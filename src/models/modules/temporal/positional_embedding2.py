import torch
from torch import nn


class PositionalEmbeddingCat(nn.Module):
    def __init__(self, sequence_length, hidden_dim, groups):
        super().__init__()
        self.groups = groups
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.register_buffer('pe', self._create_positional_encoding(sequence_length, hidden_dim, groups))  # (1, out_channels, max_len)

    def _create_positional_encoding(self, sequence_length, hidden_dim, groups):
        """
        Create positional encoding matrix.

        Returns:
            torch.Tensor: Positional encoding matrix of shape (1, out_channels, max_len).

        """
        assert self.out_dim % 2 == 0, "'num_channels // groups' must be a multiple of 2 (cos/sin)"
        pe = torch.zeros(sequence_length, hidden_dim)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.Tensor([10000.0])) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.t()[None, None, ...]  # (1, 1, dim, max_len)
        pe = pe.expand(-1, groups, -1, -1)  # (1, groups, dim, max_len)
        return pe.contiguous()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.groups, -1, self.sequence_length)
        return torch.cat((x, self.pe), dim=-2).reshape(batch_size, -1, self.sequence_length)
