import polars as pl
import torch
from torch import IntTensor
from torch.utils.data import Dataset


class TrainDataset(Dataset):

    def __init__(self, dataset: dict[str, list[int] | list[list[int]]]):
        """
        Train dataset to be consumed by DataLoader
        :param dataset: output of `initialize_train_data` defined below
        """
        self._user_idx = torch.tensor(
            dataset.pop("user_idx"),
            dtype=torch.int,
        )
        self._interacted_items = torch.tensor(
            dataset.pop("interacted_items"),
            dtype=torch.int,
        )
        self._positive_target = torch.tensor(
            dataset.pop("positive_target"),
            dtype=torch.int,
        )
        self._negative_target = torch.tensor(
            dataset.pop("negative_target"),
            dtype=torch.int,
        )

    def __len__(self):
        return len(self._user_idx)

    def __getitem__(self, idx):
        return (
            self._user_idx[idx],
            self._interacted_items[idx],
            self._positive_target[idx],
            self._negative_target[idx],
        )


class ValidationDataset(Dataset):

    def __init__(self, dataset: dict[str, list[int] | list[list[int]]]):
        """
        Validation dataset to be consumed by DataLoader
        :param dataset: input of TrainDataset input, except for `negative_target` which is unnecessary for validation
        """
        self._user_idx = torch.tensor(
            dataset["user_idx"],
            dtype=torch.int,
        )
        self._interacted_items = torch.tensor(
            dataset["interacted_items"],
            dtype=torch.int,
        )
        self._positive_target = torch.tensor(
            dataset["target"],
            dtype=torch.int,
        )

    def __len__(self):
        return len(self._user_idx)

    def __getitem__(self, idx):
        return (
            self._user_idx[idx],
            self._interacted_items[idx],
            self._positive_target[idx],
        )


def initialize_train_data(
    data: pl.DataFrame,
    negative_sample_size: int,
    max_history_length: int,
) -> dict[str, list[int] | list[list[int]]]:
    """
    Method to sample a positive target from user history, negative samples from pre-sampled pool of negative samples,
    pad user history of variable length with padding index and return result as dictionary to be passed as parameter
    of TrainDataset instance definition
    :param data: dataframe with column ('user_idx', 'interacted_items', 'negative_sample_pool', 'padding')
    :param negative_sample_size: number of negative samples
    :param max_history_length: length of padded history
    :return: dictionary of processed data
    """
    return (
        data.with_columns(
            # sample a positive target from user history
            pl.col("interacted_items").list.sample(1).alias("positive_target"),
            # sample negative samples from pre-sampled pool of negative samples
            pl.col("negative_sample_pool")
            .list.sample(negative_sample_size)
            .alias("negative_target"),
        )
        .with_columns(
            # pad user history of variable length with padding index
            pl.col("interacted_items")
            .list.set_difference("positive_target")
            .list.concat("padding")
            .list.slice(offset=0, length=max_history_length),
            # flatten positive target column
            pl.col("positive_target").list.first(),
        )
        .drop(["negative_sample_pool", "padding"])
        .to_dict(as_series=False)
    )


def load_batch(batch: list[IntTensor], device: str, training=True) -> list[IntTensor]:
    """
    Simple routine to register every data on designated device
    :param batch: list of IntTensors
    :param device: device to register every input tensor
    :param training: if training, register negative target as well
    :return: registered input data
    """
    if training:
        user_idx, interacted_items, positive_target, negative_target = batch
        user_idx = user_idx.to(device)
        interacted_items = interacted_items.to(device)
        positive_target = positive_target.to(device)
        negative_target = negative_target.to(device)
        return [user_idx, interacted_items, positive_target, negative_target]
    else:
        user_idx, interacted_items, target = batch
        user_idx = user_idx.to(device)
        interacted_items = interacted_items.to(device)
        target = target.to(device)
        return [user_idx, interacted_items, target]
