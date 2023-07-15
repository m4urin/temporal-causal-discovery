import torch
import torch.nn as nn
from src.models.modules.temporal.temporal_block import TemporalBlock
from src.models.modules.temporal.temporal_module import TemporalModule


class BaseTCN(TemporalModule):
    """
    A Temporal Convolutional Network (TCN) model that consists of a stack of Temporal Blocks.
    The TCN is designed to take in temporal sequences of data and produce a prediction for each time step.

    Args:
        in_channels (int): The number of input channels for the first Temporal Block.
        out_channels (int): The number of output channels for the last Conv1d layer.
        hidden_dim (int): The number of hidden channels for each Temporal Block.
        kernel_size (int): The kernel size for each Temporal Block.
        n_blocks (int, optional): The number of Temporal Blocks in the TCN (default: 1).
        n_layers_per_block (int, optional): The number of layers in each Temporal Block (default: 1).
        groups (int, optional): The number of groups for each Conv1d layer (default: 1).
        dropout (float, optional): The dropout probability for each Temporal Block (default: 0.0).
    """
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, n_blocks=1,
                 n_layers_per_block=1, groups=1, dropout=0.0):
        super().__init__(in_channels, out_channels, groups,
                         receptive_field=(2 ** n_blocks - 1) * n_layers_per_block * (kernel_size - 1) + 1)
        assert hidden_dim % groups == 0, "'hidden_dim' should be a multiple of 'groups'"

        # Calculate the total number of layers in the TCN
        self.n_total_layers = n_blocks * n_layers_per_block + 1

        # Initialize the activation and dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Create the stack of Temporal Blocks
        self.temporal_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.temporal_blocks.append(TemporalBlock(
                in_channels=in_channels if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=2 ** i,
                groups=groups,
                n_layers=n_layers_per_block,
                dropout=dropout)
            )

        # Create the final prediction layer
        self.predictions = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=1,
            groups=groups
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TCN model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, out_channels, seq_len).
        """
        # Pass the input through each Temporal Block in the TCN
        for layer in self.temporal_blocks:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Pass the output through the final prediction layer
        return self.predictions(x)
