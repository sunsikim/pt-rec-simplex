import torch
import torch.nn.functional as F
from torch import nn, Tensor, IntTensor


class SimpleXModel(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_dims: int,
        dropout_ratio: float,
        history_weight: float,
    ):
        """
        Implementation of model explained in https://arxiv.org/abs/2109.12613
        :param num_users: number of unique user embeddings to be estimated
        :param num_items: number of unique item embeddings in the model(including padding index embedding)
        :param num_dims: dimension of user(item) embeddings
        :param dropout_ratio: dropout ratio over augmented user embedding
        :param history_weight: weight on pooled user history embedding in user-history linear combination
        """
        super(SimpleXModel, self).__init__()
        self.user_embeddings = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=num_dims,
        )
        self.item_embeddings = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=num_dims,
            padding_idx=0,  # embedding corresponds to index=0 will be zero vector
        )
        self.history_linear_map = nn.Linear(
            in_features=num_dims,
            out_features=num_dims,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout_ratio)
        self._g = 1 - history_weight

    def forward(
        self,
        user_idx: IntTensor,
        interacted_items: IntTensor,
        target_idx: IntTensor,
    ) -> Tensor:
        """
        Implementation of SimpleX cosine similarity calculation logic
        :param user_idx: 1D Tensor of user indices
        :param interacted_items: 2D square Tensor of item indices padded with padding index(0)
        :param target_idx: 2D columnar Tensor of item indices to calculate cosine user-item cosine similarity
        :return: BATCH_SIZE or (BATCH_SIZE, NEGATIVE_SAMPLE_SIZE) shaped Tensor
        """
        assert user_idx.dim() == 1 and target_idx.dim() == 2
        # aggregate user history
        history = self._pool_user_history(interacted_items)
        history = self.history_linear_map(history)
        # calculate user embedding
        user_vector = self.user_embeddings(user_idx) * self._g + history * (1 - self._g)
        user_vector = self.dropout(user_vector)
        user_vector = F.normalize(user_vector, dim=1).unsqueeze(
            dim=-1
        )  # (BATCH_SIZE, NUM_DIMS, 1)
        # fetch item embeddings
        item_vector = self.item_embeddings(
            target_idx
        )  # (BATCH_SIZE, {1, NEGATIVE_SAMPLE_SIZE}, NUM_DIMS)
        item_vector = F.normalize(item_vector, dim=-1)
        # squeeze batch matrix multiplication result to get Tensor of cosine similarities
        return torch.bmm(item_vector, user_vector).squeeze()

    def predict(self, user_idx: IntTensor, interacted_items: IntTensor) -> Tensor:
        """
        Given user_idx with corresponding select history, calculates cosine similarity over every item
        :param user_idx: 1D Tensor of user indices
        :param interacted_items: 2D square Tensor of item indices padded with padding index(0)
        :return: (BATCH_SIZE, NUM_ITEMS) shaped Tensor
        """
        assert user_idx.dim() == 1
        with torch.no_grad():
            # aggregate user history
            history = self._pool_user_history(interacted_items)
            history = self.history_linear_map(history)
            # calculate user embedding
            user_vector = self.user_embeddings(user_idx) * self._g + history * (
                1 - self._g
            )
            user_vector = F.normalize(user_vector, dim=1)
            # fetch whole item embedding
            item_vector = F.normalize(self.item_embeddings.weight, dim=1)
            item_vector = item_vector.transpose(dim0=0, dim1=1)
            return user_vector @ item_vector

    def _pool_user_history(self, interacted_items: IntTensor) -> Tensor:
        """
        Among three different types of history pooling, this code implements average pooling.
        Since padding_idx parameter is specified, it corresponds to zero vector in item_embedding.
        As a result, output can be directly summed over column-wise dimension(dim=1).
        Note that shape of (1) = (BATCH_SIZE, NUM_DIMS) and shape of (2) = (BATCH_SIZE, 1)
        """
        count_none_paddings = torch.ne(input=interacted_items, other=0).sum(dim=1)
        pooled_embeddings = self.item_embeddings(interacted_items).sum(dim=1)  # (1)
        pooled_embeddings /= count_none_paddings.unsqueeze(dim=-1)  # (2)
        return pooled_embeddings


class CosineContrastiveLoss(nn.Module):

    def __init__(self, margin: float, negative_weight: int | float):
        super(CosineContrastiveLoss, self).__init__()
        if not -1.0 < margin < 1.0:
            raise ValueError("margin of cosine similarity does not fall into (-1, 1)")
        elif negative_weight < 0:
            raise ValueError(
                "weight on loss calculated from negative samples cannot be negative"
            )
        self._margin = margin
        self._negative_weight = float(negative_weight)

    def forward(self, y_pred, y_true=None):
        """
        calculate CCL as explained in SimpleX paper
        :param y_pred: (batch_size, 1 + negative_sample_size) shaped tensor of cosine similarities
        :param y_true: dummy placeholder of loss function to comply Pytorch convention
        :return: mean-reduced cosine contrastive loss
        """
        positive_loss = torch.relu(1 - y_pred[:, 0])
        negative_loss = torch.relu(y_pred[:, 1:] - self._margin)
        negative_loss = torch.mean(negative_loss, dim=1) * self._negative_weight
        loss = positive_loss + negative_loss  # vector of length batch_size
        return torch.mean(loss)
