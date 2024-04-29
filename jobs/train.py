import polars as pl
import jobs.config as config
import pathlib
import logging
import numpy as np
import torch
import os
from jobs.config import ExecutableJobs
from simplex.model import SimpleXModel, CosineContrastiveLoss
from simplex.trainer import SimpleXTrainer


class TrainJob(ExecutableJobs):

    def __init__(self):
        super(TrainJob, self).__init__(job_name="train")
        self._local_dir = pathlib.Path(config.LOCAL_DIR)
        self._data_dir = pathlib.Path(config.DATA_DIR)

    def execute(self):
        logging.info("Load data from preprocess job")
        ratings = pl.read_parquet(self._data_dir.joinpath(config.RATINGS_FILE_NAME))
        mid2idx = pl.read_parquet(
            self._data_dir.joinpath(config.MOVIE_INDEX_MAP_FILE_NAME)
        )
        num_users = ratings.height
        num_items = mid2idx.height

        logging.info("Split data into train, validation data")
        train_data, validation_data = self._split_data(ratings)

        logging.info("Define model and corresponding trainer")
        model = SimpleXModel(
            num_users=num_users,
            num_items=num_items + 1,  # extra slot for padding index
            num_dims=config.EMBEDDING_DIM,
            dropout_ratio=0.3,
            history_weight=0.5,
        )
        trainer = self._define_model_trainer(model, train_data, validation_data)

        logging.info("Train model")
        trainer.execute()

    def _split_data(self, ratings: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        validation_data = (
            ratings.filter(pl.col("interacted_items").list.len() >= 3)
            .sample(fraction=config.VALIDATION_USER_RATIO)
            .with_columns(
                pl.col("interacted_items").list.first().alias("target"),
                pl.col("interacted_items").list.slice(offset=1).alias("remainder"),
            )
            .drop(["interacted_items", "negative_sample_pool"])
        )
        train_data = (
            ratings.join(validation_data, on="user_idx", how="left")
            .with_columns(
                pl.when(pl.col("remainder").is_null())
                .then("interacted_items")
                .otherwise("remainder")
                .alias("interacted_items")
            )
            .drop(["target", "remainder"])
        )

        # validation data doesn't need to be sampled afterward, so pad the history and return
        validation_data = (
            self._attach_padding_column(validation_data)
            .rename({"remainder": "interacted_items", "target": "positive_target"})
            .with_columns(
                pl.col("interacted_items")
                .list.concat("padding")
                .list.slice(offset=0, length=config.HISTORY_MAX_LEN)
            )
            .drop("padding")
        )
        train_data = self._attach_padding_column(train_data)
        return train_data, validation_data

    def _define_model_trainer(
        self,
        model: SimpleXModel,
        train_data: pl.DataFrame,
        validation_data: pl.DataFrame,
    ) -> SimpleXTrainer:
        gpu_id = os.environ.get("GPU_ID", "0")
        optimizer = torch.optim.Adam(params=model.parameters())
        loss_fn = CosineContrastiveLoss(
            margin=config.LOSS_NEGATIVE_MARGIN,
            negative_weight=config.LOSS_NEGATIVE_WEIGHT,
        )
        return SimpleXTrainer(
            device=f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu",
            local_dir=self._local_dir,
            checkpoint_file=config.CHECKPOINT_FILE_NAME,
            metric_log_file=config.TRAIN_LOG_FILE,
            model=model,
            loss_fn=loss_fn,
            train_data=train_data,
            validation_data=validation_data,
            optimizer=optimizer,
            batch_size=config.BATCH_SIZE,
            num_epochs=config.MAX_EPOCH,
            negative_sample_size=config.NEGATIVE_SAMPLE_SIZE,
            max_history_length=config.HISTORY_MAX_LEN,
        )

    def _attach_padding_column(self, data: pl.DataFrame) -> pl.DataFrame:
        padding_value = np.zeros(shape=(data.height, config.HISTORY_MAX_LEN), dtype=int)
        return data.insert_column(index=-1, column=pl.Series("padding", padding_value))
