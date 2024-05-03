import polars as pl
import torch
import pathlib
import time
import logging
import numpy as np
import jobs.config as config
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from simplex.model import CosineContrastiveLoss, SimpleXModel
from simplex.data import (
    TrainDataset,
    ValidationDataset,
    load_batch,
    initialize_train_data,
)


class SimpleXTrainer:

    def __init__(
        self,
        device: str,
        experiment_prefix: str,
        local_dir: pathlib.Path,
        checkpoint_file: str,
        model: SimpleXModel,
        loss_fn: CosineContrastiveLoss,
        train_data: pl.DataFrame,
        validation_data: pl.DataFrame,
        optimizer: torch.optim.Adam,
        batch_size: int,
        num_epochs: int,
        negative_sample_size: int,
        max_history_length: int,
        num_validation_products: int,
    ):
        self.device = device
        self.local_dir = local_dir
        self.checkpoint_path = local_dir.joinpath(checkpoint_file)
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.validation_data = validation_data
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.negative_sample_size = negative_sample_size
        self.max_history_length = max_history_length
        self.experiment_prefix = experiment_prefix
        self._best_validation_metric = 0.0
        self._not_improved_since = 0
        self._max_down_streak = 10
        self._k = num_validation_products
        self._validation_loader = DataLoader(
            dataset=ValidationDataset(validation_data.to_dict(as_series=False)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

    def execute(self):
        experiment_name = f"runs/{self.experiment_prefix}/"
        experiment_name += f"dim={config.EMBEDDING_DIM},"
        experiment_name += f"margin={config.LOSS_NEGATIVE_MARGIN},"
        experiment_name += f"batch={config.BATCH_SIZE},"
        experiment_name += f"negsize={config.NEGATIVE_SAMPLE_SIZE}"
        writer = SummaryWriter(experiment_name)
        for epoch_idx in range(1, self.num_epochs + 1):
            # iterate one epoch
            epoch_start_at = time.time()
            training_loss = self._train_one_epoch()
            validation_metrics = self._calculate_validation_metrics()
            elapsed_seconds = int(time.time() - epoch_start_at)

            # record epoch metrics
            recall, precision, hr, ndcg = validation_metrics
            epoch_result = [epoch_idx, elapsed_seconds, training_loss]
            writer.add_scalar("Loss/CCL", training_loss, epoch_idx)
            writer.add_scalar(f"Metrics/Recall@{self._k}", recall, epoch_idx)
            writer.add_scalar(f"Metrics/Precision@{self._k}", precision, epoch_idx)
            writer.add_scalar(f"Metrics/HR@{self._k}", hr, epoch_idx)
            writer.add_scalar(f"Metrics/nDCG@{self._k}", ndcg, epoch_idx)
            log_msg = [
                f"epoch {epoch_idx:>3}",
                f"({elapsed_seconds}s)",
                f"loss = {training_loss:.4f}",
                f"recall@{self._k} = {recall:.4f}",
                f"precision@{self._k} = {precision:.4f}",
                f"hr@{self._k} = {hr:.4f}",
                f"nDCG@{self._k} = {ndcg:.4f}",
            ]
            logging.info(" | ".join(log_msg))

            # determine early stopping
            if self._evaluate_early_stopping_criteria(recall):
                break

    def _train_one_epoch(self) -> float:
        epoch_loss = 0
        train_data = initialize_train_data(
            data=self.train_data,
            negative_sample_size=self.negative_sample_size,
            max_history_length=self.max_history_length,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=TrainDataset(train_data),
            batch_size=self.batch_size,
            drop_last=True,
        )
        self.model.train(True)

        for batch_idx, batch in enumerate(train_loader):
            user_idx, interacted_items, positive_target, negative_target = load_batch(
                batch=batch, device=self.device, training=True
            )
            positive_target = positive_target.unsqueeze(dim=-1)

            # calculate loss
            self.optimizer.zero_grad()
            positive_cossims = self.model(user_idx, interacted_items, positive_target)
            negative_cossims = self.model(user_idx, interacted_items, negative_target)
            cossims = torch.cat(
                tensors=(positive_cossims.unsqueeze(dim=-1), negative_cossims), dim=1
            )
            batch_loss = self.loss_fn(cossims)

            # calculate gradient
            batch_loss.backward()
            epoch_loss += batch_loss.item()

            # back propagate loss
            self.optimizer.step()

        return epoch_loss / (batch_idx + 1)

    def _calculate_validation_metrics(self) -> tuple[float, float, float, float]:
        recall = []
        precision = []
        hit_ratio = []
        ndcg = []
        self.model.eval()
        with torch.no_grad():
            for batch in self._validation_loader:
                user_idx, interacted_items, target = load_batch(
                    batch, device=self.device, training=False
                )

                # calculate cosine similarities over every item
                cossims = self.model.predict(user_idx, interacted_items)

                # select top k items among product pool and calculate validation metrics
                pred = torch.argsort(cossims, descending=True)[:, : self._k].cpu()
                pred = pred.numpy()
                true = target.cpu().numpy()
                recall.append(self.calculate_batch_recall(pred, true))
                precision.append(self.calculate_batch_precision(pred, true))
                hit_ratio.append(self.calculate_batch_hr(pred, true))
                ndcg.append(self.calculate_batch_ndcg(pred, true))

        return np.mean(recall), np.mean(precision), np.mean(hit_ratio), np.mean(ndcg)

    def _evaluate_early_stopping_criteria(self, criteria: float) -> bool:
        early_stop = False
        if self._best_validation_metric <= criteria:
            self._not_improved_since = 0
            self._best_validation_metric = criteria
            checkpoint = self.model.state_dict()
            torch.save(checkpoint, self.checkpoint_path)
        elif self._not_improved_since < self._max_down_streak:
            self._not_improved_since += 1
        else:
            logging.info(
                f"Validation metric has not improved for {self._max_down_streak} epochs, so early stopping"
            )
            early_stop = True
        return early_stop

    @staticmethod
    def calculate_batch_precision(top_k_items: np.array, target: np.array) -> float:
        """
        proportion of recommended items in the top-k set that are relevant
        """
        num_hits = [len(np.intersect1d(x[0], x[1])) for x in zip(top_k_items, target)]
        batch_precision = np.array(num_hits) / top_k_items.shape[1]
        return np.mean(batch_precision)

    @staticmethod
    def calculate_batch_recall(top_k_items: np.array, target: np.array) -> float:
        """
        proportion of relevant items found in the top-k recommendations
        """
        num_selected = np.sum(target != 0, axis=1)
        num_hits = [len(np.intersect1d(x[0], x[1])) for x in zip(top_k_items, target)]
        batch_recall = np.array(num_hits) / num_selected
        return np.mean(batch_recall)

    @staticmethod
    def calculate_batch_hr(top_k_items: np.array, target: np.array) -> float:
        """
        proportion of users which gets recommendation that contains one of relevant items
        """
        num_hits = [len(np.intersect1d(x[0], x[1])) for x in zip(top_k_items, target)]
        is_hit_user = np.array(num_hits) > 0
        hit_ratio = np.sum(is_hit_user) / top_k_items.shape[0]
        return hit_ratio

    @staticmethod
    def calculate_batch_ndcg(top_k_items: np.array, target: np.array) -> float:
        """
        metric scaling from 0 to 1 that rewards result with relevant items placed in higher priority
        """
        binary_pred = []
        for top_k_item, true in zip(top_k_items, target):
            binary_pred.append([1 if item in true else 0 for item in top_k_item])
        binary_pred = np.array(binary_pred)
        denominator = np.log2(1 + np.arange(start=1, stop=binary_pred.shape[1] + 1))
        dcg = np.sum(binary_pred / denominator, axis=1)
        idcg = (1 / denominator).sum()
        return np.mean(dcg / idcg)
