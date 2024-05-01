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
    """
    Train a SimpleX model on a ML-1M dataset and explore various hyperparameter combinations
    """

    def __init__(self, job_name: str = "train"):
        super(TrainJob, self).__init__(job_name=job_name)
        self._local_dir = pathlib.Path(config.LOCAL_DIR)
        self._data_dir = pathlib.Path(config.DATA_DIR)
        self._job_name = job_name

    def execute(self):
        logging.info("Load data from preprocess job")
        ratings = pl.read_parquet(self._data_dir.joinpath(config.RATINGS_FILE_NAME))
        mid2idx = pl.read_parquet(
            self._data_dir.joinpath(config.MOVIE_INDEX_MAP_FILE_NAME)
        )
        num_users = ratings.height
        num_items = mid2idx.height

        logging.info("Split data into train, validation data")
        train_data, validation_data, _ = split_data(ratings)

        logging.info("Define model and corresponding trainer")
        model = SimpleXModel(
            num_users=num_users,
            num_items=num_items + 1,  # extra slot for padding index
            num_dims=config.EMBEDDING_DIM,
            dropout_ratio=0.3,
            history_weight=0.5,
        )
        trainer = define_model_trainer(
            model=model,
            job_name=self._job_name,
            local_dir=self._local_dir,
            train_data=train_data,
            validation_data=validation_data,
        )

        logging.info("Train model")
        trainer.execute()


def split_data(ratings: pl.DataFrame) -> list[pl.DataFrame]:
    num_holdout_users = int(ratings.height * config.HOLDOUT_RATIO)
    num_train_users = ratings.height - num_holdout_users
    validation_feed = ratings.slice(
        offset=num_train_users,
        length=num_holdout_users // 2,
    )
    validation_feed, validation_data = _split_user_history(validation_feed)
    test_feed = ratings.slice(
        offset=num_train_users + num_holdout_users // 2,
        length=num_holdout_users // 2,
    )
    test_feed, test_data = _split_user_history(test_feed)
    train_data = ratings.slice(offset=0, length=num_train_users)
    train_data = _attach_padding_column(
        pl.concat([train_data, validation_feed, test_feed])
    )
    return [train_data, validation_data, test_data]


def define_model_trainer(
    job_name: str,
    local_dir: pathlib.Path,
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
        experiment_prefix=job_name,
        local_dir=local_dir,
        checkpoint_file=config.CHECKPOINT_FILE_NAME,
        model=model,
        loss_fn=loss_fn,
        train_data=train_data,
        validation_data=validation_data,
        optimizer=optimizer,
        batch_size=config.BATCH_SIZE,
        num_epochs=config.MAX_EPOCH,
        negative_sample_size=config.NEGATIVE_SAMPLE_SIZE,
        max_history_length=config.HISTORY_MAX_LEN,
        num_validation_products=config.NUM_VALIDATION_PRODUCTS,
    )


def _split_user_history(data: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    holdout_data = (
        data.with_columns(
            (pl.col("interacted_items").list.len() * config.HOLDOUT_RATIO)
            .cast(pl.Int32)
            .alias("num_targets")
        )
        .with_columns(
            pl.col("interacted_items")
            .list.slice(offset=0, length="num_targets")
            .alias("target"),
            pl.col("interacted_items").list.slice(offset="num_targets"),
        )
        .drop("num_targets")
    )
    feed_data = holdout_data.drop("target")
    holdout_data = holdout_data.drop("negative_sample_pool")
    holdout_data = (
        _attach_padding_column(holdout_data)
        .with_columns(
            pl.col("interacted_items")
            .list.concat("padding")
            .list.slice(offset=0, length=config.HISTORY_MAX_LEN)
        )
        .with_columns(
            pl.col("target")
            .list.concat("padding")
            .list.slice(offset=0, length=config.HISTORY_MAX_LEN)
        )
        .drop("padding")
    )
    return feed_data, holdout_data


def _attach_padding_column(data: pl.DataFrame) -> pl.DataFrame:
    padding_value = np.zeros(shape=(data.height, config.HISTORY_MAX_LEN), dtype=int)
    return data.insert_column(index=-1, column=pl.Series("padding", padding_value))
