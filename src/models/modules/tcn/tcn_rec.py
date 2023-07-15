import torch.nn as nn
from src.models.modules.temporal.temporal_block import TemporalBlock
from src.models.modules.temporal.temporal_module import TemporalModule


class RecTCN(TemporalModule):
    """
    A Recurrent Temporal Convolutional Network (Rec-TCN) model that consists of a first Temporal Block
    followed by a series of Recurrent Blocks. The Rec-TCN is designed to take in temporal sequences of data
    and produce a prediction for each time step.

    This version of the Rec-TCN has much fewer number of parameters compared to a general TCN when the number
    of layers is large.

    Args:
        in_channels (int): The number of input channels for the first Temporal Block.
        out_channels (int): The number of output channels for the last Conv1d layer.
        hidden_dim (int): The number of hidden channels for each Temporal Block.
        kernel_size (int): The kernel size for each Temporal Block.
        n_blocks (int, optional): The number of Recurrent Blocks in the Rec-TCN (default: 3).
        n_layers_per_block (int, optional): The number of layers in each Temporal Block (default: 1).
        groups (int, optional): The number of groups for each Conv1d layer (default: 1).
        dropout (float, optional): The dropout probability for each Temporal Block (default: 0.0).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_dim,
                 kernel_size,
                 n_blocks=3,
                 n_layers_per_block=1,
                 groups=1,
                 dropout=0.0):
        super().__init__(in_channels, out_channels, groups,
                         receptive_field=(2 ** n_blocks - 1) * n_layers_per_block * (kernel_size - 1) + 1)

        self.n_blocks = n_blocks

        assert self.n_blocks >= 3, 'if n_blocks is smaller than 2, please use a regular TCN'

        # Define the first TCN block
        self.first_block = TemporalBlock(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=1,
            groups=groups,
            n_layers=n_layers_per_block,
            dropout=dropout
        )

        # Define the recurrent block
        self.recurrent = TemporalBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=1,
            groups=groups,
            n_layers=n_layers_per_block,
            dropout=dropout
        )

        # Define the dropout and ReLU layers
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Create the final prediction layer
        self.predictions = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=1,
            groups=groups
        )

    def forward(self, x):
        """
        Forward pass through the Rec-TCN model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, out_channels, seq_len).
        """
        # Pass input tensor through the first TCN block
        x = self.first_block(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Pass input tensor through the residual blocks
        for i in range(self.n_blocks - 1):
            # Update the dilation factor for the Recurrent Block
            self.recurrent.dilation = 2 ** (i + 1)

            # Pass input tensor through the Recurrent Block
            x = self.recurrent(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Pass the output through the final prediction layer
        return self.predictions(x)

