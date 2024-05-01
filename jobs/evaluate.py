import numpy as np
import polars as pl
import torch
import jobs.config as config
import pathlib
import logging
from torch.utils.data import DataLoader
from jobs.config import ExecutableJobs
from jobs.train import split_data, define_model_trainer
from simplex.model import SimpleXModel
from simplex.trainer import SimpleXTrainer
from simplex.data import ValidationDataset


class EvaluateJob(ExecutableJobs):
    """
    Train a SimpleX model using ML-20M data with the best hyperparameter combination and evaluate model on test data
    """

    def __init__(self, job_name: str = "evaluate"):
        super(EvaluateJob, self).__init__(job_name=job_name)
        self._local_dir = pathlib.Path(config.LOCAL_DIR)
        self._data_dir = pathlib.Path(config.DATA_DIR)
        self._job_name = job_name

    def execute(self):
        logging.info("Load data from preprocess job")
        ratings = pl.read_parquet(self._data_dir.joinpath(config.RATINGS_FILE_NAME))
        midx2id = pl.read_parquet(
            self._data_dir.joinpath(config.MOVIE_INDEX_MAP_FILE_NAME)
        )
        num_users = ratings.height
        num_items = midx2id.height

        logging.info("Split data into train, validation data")
        train_data, validation_data, test_data = split_data(ratings)

        logging.info("Define model and corresponding trainer")
        model = SimpleXModel(
            num_users=num_users,
            num_items=num_items + 1,
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

        logging.info("Make prediction on test data")
        model = self._load_trained_model(num_users, num_items)
        pred = self._make_prediction(model, test_data)

        logging.info("Evaluate prediction result")
        test_data = test_data.to_dict(as_series=False)
        true = np.array(test_data["target"])
        precision = SimpleXTrainer.calculate_batch_precision(pred, true)
        recall = SimpleXTrainer.calculate_batch_recall(pred, true)
        hr = SimpleXTrainer.calculate_batch_hr(pred, true)
        ndcg = SimpleXTrainer.calculate_batch_ndcg(pred, true)

        logging.info("Convert movie index to original movie id")
        midx2id = dict(midx2id.to_pandas().values)
        interacted_items = [
            [midx2id[item] for item in items if item != 0]
            for items in test_data["interacted_items"]
        ]
        target = [
            [midx2id[item] for item in items if item != 0]
            for items in test_data["target"]
        ]
        prediction = [[midx2id[item] for item in items if item != 0] for items in pred]
        result = pl.DataFrame(
            {
                "user_idx": test_data["user_idx"],
                "interacted_items": interacted_items,
                "target": target,
                "prediction": prediction,
            }
        )

        logging.info("Save evaluation result")
        result.write_parquet(
            self._local_dir.joinpath(config.RECOMMENDATION_RESULT_FILE_NAME)
        )
        with open(
            self._local_dir.joinpath(config.METRIC_RESULT_FILE_NAME), "a"
        ) as file:
            file.write(f"precision@{config.NUM_VALIDATION_PRODUCTS},")
            file.write(f"recall@{config.NUM_VALIDATION_PRODUCTS},")
            file.write(f"hr@{config.NUM_VALIDATION_PRODUCTS},")
            file.write(f"ndcg@{config.NUM_VALIDATION_PRODUCTS}\n")
            file.write(f"{precision},{recall},{hr},{ndcg}")

    def _load_trained_model(self, num_users, num_items) -> SimpleXModel:
        model = SimpleXModel(
            num_users=num_users,
            num_items=num_items + 1,
            num_dims=config.EMBEDDING_DIM,
            dropout_ratio=0.3,
            history_weight=0.5,
        )
        ckpt = torch.load(self._local_dir.joinpath(config.CHECKPOINT_FILE_NAME))
        model.load_state_dict(ckpt)
        return model.to("cpu")

    def _make_prediction(
        self, model: SimpleXModel, test_data: pl.DataFrame
    ) -> np.array:
        predictions = []
        model.eval()
        test_loader = DataLoader(
            dataset=ValidationDataset(test_data.to_dict(as_series=False)),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        )
        with torch.no_grad():
            for batch in test_loader:
                user_idx, interacted_items, target = batch
                cossims = model.predict(user_idx, interacted_items)
                top_k_items = torch.argsort(cossims, dim=1, descending=True)
                top_k_items = top_k_items[:, : config.NUM_VALIDATION_PRODUCTS]
                predictions.append(top_k_items.numpy())
        return np.vstack(predictions)
