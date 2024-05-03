import jobs
import numpy as np
from simplex.trainer import SimpleXTrainer

pred = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
true = [[3, 6, 0, 0], [1, 4, 5, 7], [1, 2, 8, 0]]


def test_precision_calculation():
    metric = SimpleXTrainer.calculate_batch_precision(np.array(pred), np.array(true))
    assert np.isclose(metric, 0.2)  # (0.2 + 0.4 + 0) / 3


def test_recall_calculation():
    metric = SimpleXTrainer.calculate_batch_recall(np.array(pred), np.array(true))
    assert np.isclose(metric, 1 / 3)  # (0.5 + 0.5 + 0) / 3


def test_hr_calculation():
    metric = SimpleXTrainer.calculate_batch_hr(np.array(pred), np.array(true))
    assert np.isclose(metric, 2 / 3)  # (1 + 1 + 0) / 3


def test_ndcg_calculation():
    metric = SimpleXTrainer.calculate_batch_ndcg(np.array(pred), np.array(true))
    binary_pred = [[0, 0, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0]]
    binary_pred = np.array(binary_pred)
    denominator = np.log2(1 + np.arange(1, 6))
    dcg = (binary_pred / denominator).sum(axis=1)
    idcg = (1 / denominator).sum()
    assert np.isclose(metric, np.mean(dcg / idcg))
