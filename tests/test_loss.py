import pytest
import torch
import numpy as np
from simplex.loss import CosineContrastiveLoss


def test_invalid_margin():
    with pytest.raises(ValueError):
        invalid_margin = 1.1
        loss_fn = CosineContrastiveLoss(invalid_margin, 100)


def test_invalid_negative_weight():
    with pytest.raises(ValueError):
        invalid_negative_weight = -100
        loss_fn = CosineContrastiveLoss(0.5, invalid_negative_weight)


def test_calculated_value():
    margin = 0.5
    negative_weight = 1.5
    loss_fn = CosineContrastiveLoss(margin, negative_weight)
    cosine_similarities = torch.Tensor(
        [
            [0.1, 0.3, 0.5, 0.7],
            [0.2, 0.4, 0.6, 0.8],
            [0.3, 0.5, 0.7, 0.9],
        ]
    )
    calculated_loss = loss_fn(cosine_similarities).numpy()
    positive_losses = 1 - np.array([0.1, 0.2, 0.3])
    negative_losses = np.array([0.0667, 0.1333, 0.2]) * negative_weight
    answer_loss = np.mean(positive_losses + negative_losses)
    assert np.isclose(answer_loss, calculated_loss)
