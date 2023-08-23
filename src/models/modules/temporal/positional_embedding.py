import torch
from torch import nn

from src.models.modules.temporal.temporal_module import TemporalModule


class PositionalEmbedding(TemporalModule):
    """
    Positional Embedding module for temporal data.

    This module adds positional encoding to the input data to capture the sequential information.
    It creates a positional encoding matrix and applies it to the input tensor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        max_len (int): Maximum sequence length.
        groups (int): Number of groups.

    Attributes:
        pe (torch.Tensor): Positional encoding tensor of shape (1, out_channels, max_len).
        embeddings (torch.Parameter): Learnable parameter for additional embeddings of shape (1, out_channels, 1).
        down_sample (nn.Conv1d or None): Down-sampling convolutional layer, if in_channels != out_channels.
        norm (nn.GroupNorm): Group normalization layer.

    """

    def __init__(self, in_channels: int, out_channels: int, max_len: int, groups: int):
        super().__init__(in_channels, out_channels, groups)
        assert self.out_dim % 2 == 0, "'num_channels // groups' must be a multiple of 2 (cos/sin)"
        self.max_len = max_len
        self.register_buffer('pe', self._create_positional_encoding())  # (1, out_channels, max_len)
        # TODO remove max_len
        self.embeddings = nn.Parameter(0.1 * torch.randn(1, self.out_channels, max_len))  # (1, out_channels, 1)
        self.down_sample = None
        self.norm = None

        if in_channels > 0:
            if in_channels != out_channels:
                self.down_sample = nn.Conv1d(
                    in_channels=self.in_dim,
                    out_channels=self.out_dim,
                    kernel_size=1,
                    groups=1,
                    bias=False  # No need for bias, as it will be captured in the group_norm
                )
            self.norm = nn.GroupNorm(num_channels=self.out_dim, num_groups=1)

    def _create_positional_encoding(self):
        """
        Create positional encoding matrix.

        Returns:
            torch.Tensor: Positional encoding matrix of shape (1, out_channels, max_len).

        """
        pe = torch.zeros(self.max_len, self.out_dim)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.out_dim, 2).float() * (-torch.log(torch.Tensor([10000.0])) / self.out_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.t()[None, None, ...]  # (1, 1, out_dim, max_len)
        pe = pe.expand(-1, self.groups, -1, -1)  # (1, groups, dim, max_len)
        pe = self.channel_view(pe)  # (1, out_channels, max_len)
        return pe

    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the PositionalEmbedding module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_len).
            use_max_len (bool): Whether to use the maximum length for positional encoding.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, seq_len).

        """
        if x is None:
            #return self.pe + self.embeddings
            return self.embeddings

        batch_size, _, seq_len = x.size()
        x = x.reshape(-1, self.in_dim, seq_len)  # (batch_size * groups, in_dim, seq_len)

        if self.down_sample is not None:
            x = self.down_sample(x)  # (batch_size * groups, out_dim, seq_len)

        x = self.norm(x)
        x = x.view(batch_size, -1, seq_len)  # (batch_size, out_channels, seq_len)

        #print(x.size(), self.embeddings.size())
        return x + self.embeddings[..., :seq_len]  ##+ self.pe[..., :seq_len]

    def get_embeddings(self):
        """
        Get the learnable embeddings.

        Returns:
            torch.Tensor: Embeddings tensor of shape (groups, out_dim).

        """
        return self.embeddings.view(self.groups, -1)

    def __repr__(self):
        s = "PositionalEmbedding(\n"
        if self.down_sample is not None:
            s += f"  (down_sample): {self.down_sample}\n"
        if self.norm is not None:
            s += f"  (norm): {self.norm}\n"
        s += f"  (embedding): Embedding(channels={self.out_dim}, groups={self.groups})\n)"
        return s


if __name__ == '__main__':
    from src.utils.pytorch import count_parameters

    pos_emb = PositionalEmbedding(8, 128, 100, 8)
    data = torch.randn(1, 8, 100)
    result = pos_emb(data)
    pos_emb = PositionalEmbedding(0, 128, 100, 8)
    print(pos_emb)
    print(count_parameters(pos_emb))
    print(pos_emb().size())
