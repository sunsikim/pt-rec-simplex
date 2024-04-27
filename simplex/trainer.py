import polars as pl
import torch
import os
import pathlib
import time
import logging
from torch.utils.data import Dataset, DataLoader
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
        local_dir: pathlib.Path,
        metric_log_file: str,
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
    ):
        self.device = device
        self.local_dir = local_dir
        self.checkpoint_path = local_dir.joinpath(checkpoint_file)
        self.metric_log_path = local_dir.joinpath(metric_log_file)
        assert self.metric_log_path.suffix == ".csv"
        if self.metric_log_path.exists():
            os.remove(self.metric_log_path)
        header = [
            "epoch_idx",
            "elapsed_seconds",
            "training_loss",
            "precision_at_5",
            "precision_at_10",
            "precision_at_20",
        ]
        with open(self.metric_log_path, "a") as file:
            file.write(",".join(header))
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.validation_data = validation_data
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.negative_sample_size = negative_sample_size
        self.max_history_length = max_history_length
        self._best_validation_metric = 0.0
        self._not_improved_since = 0
        self._max_down_streak = 5
        self._validation_loader = DataLoader(
            dataset=ValidationDataset(validation_data.to_dict(as_series=False)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

    def execute(self):
        for epoch_idx in range(1, self.num_epochs + 1):
            # iterate one epoch
            epoch_start_at = time.time()
            training_loss = self._train_one_epoch()
            validation_metrics = self._calculate_validation_metrics()
            elapsed_seconds = int(time.time() - epoch_start_at)

            # record epoch metrics
            epoch_result = [epoch_idx, elapsed_seconds, training_loss]
            epoch_result.extend(validation_metrics)
            with open(self.metric_log_path, "a") as file:
                file.write(f"\n{','.join(map(str, epoch_result))}")
            log_msg = [
                f"epoch {epoch_result[0]:>3}",
                f"({epoch_result[1]}s)",
                f"loss = {epoch_result[2]:.4f}",
                f"precision at 5 = {epoch_result[3]:.4f}, 10 = {epoch_result[4]:.4f}, 20 = {epoch_result[5]:.4f}",
            ]
            logging.info(" | ".join(log_msg))

            # determine early stopping
            if self._evaluate_early_stopping_criteria(epoch_result[-1]):
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

    def _calculate_validation_metrics(self) -> tuple[float, float, float]:
        precision_at_5 = torch.tensor([0], device=self.device)
        precision_at_10 = torch.tensor([0], device=self.device)
        precision_at_20 = torch.tensor([0], device=self.device)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self._validation_loader):
                user_idx, interacted_items, target = load_batch(
                    batch, device=self.device, training=False
                )
                target = target.unsqueeze(dim=-1)

                # calculate cosine similarities over every item
                cossims = self.model.predict(user_idx, interacted_items)

                # select top 20 items among product pool and calculate validation metrics
                top_20_items = torch.argsort(cossims, descending=True)[:, :20]
                precision_at_5 += (top_20_items[:, :5] == target).sum()
                precision_at_10 += (top_20_items[:, :10] == target).sum()
                precision_at_20 += (top_20_items == target).sum()

        denominator = (batch_idx + 1) * self.batch_size
        precision_at_5 = precision_at_5.to("cpu").item() / denominator
        precision_at_10 = precision_at_10.to("cpu").item() / denominator
        precision_at_20 = precision_at_20.to("cpu").item() / denominator
        return precision_at_5, precision_at_10, precision_at_20

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
