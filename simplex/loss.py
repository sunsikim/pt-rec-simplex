import torch
from torch import nn


class CosineContrastiveLoss(nn.Module):

    def __init__(self, margin: float, negative_weight: int | float):
        super(CosineContrastiveLoss, self).__init__()
        if not -1.0 < margin < 1.0:
            raise ValueError("margin of cosine similarity does not fall into (-1, 1)")
        elif negative_weight < 0:
            raise ValueError("weight on loss calculated from negative samples cannot be negative")
        self._margin = margin
        self._negative_weight = float(negative_weight)

    def forward(self, y_pred, y_true=None):
        """
        calculate CCL as defined in SimpleX paper
        :param y_pred: (batch_size, 1 + negative_sample_size) shaped tensor of cosine similarities
        :param y_true: dummy placeholder of loss function to comply Pytorch convention
        :return: mean-reduced cosine contrastive loss
        """
        positive_loss = torch.relu(1 - y_pred[:, 0])
        negative_loss = torch.relu(y_pred[:, 1:] - self._margin)
        negative_loss = torch.mean(negative_loss, dim=1) * self._negative_weight
        loss = positive_loss + negative_loss  # vector of length batch_size
        return torch.mean(loss)
