import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.modules.temporal.downsample import DownSample
from src.models.modules.temporal.temporal_module import TemporalModule


class TemporalBlock(TemporalModule):
    """
    A PyTorch module that implements a Temporal Block.

    This module consists of a sequence of 1D convolutional layers with dilations, followed by ReLU activations
    and dropout layers. The number of layers, kernel size, dilation, and dropout rate can be configured
    during initialization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation (int, optional): Dilation factor for the convolutional layers (default: 1).
        groups (int, optional): Number of groups to split the input and output channels into (default: 1).
        n_layers (int, optional): Number of convolutional layers in the block (default: 1).
        dropout (float, optional): Dropout rate to use between layers (default: 0.0).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        n_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__(in_channels, out_channels, groups,
                         receptive_field=n_layers * dilation * (kernel_size - 1) + 1)

        assert n_layers > 0, "Number of layers should be 1 or greater."

        # Set class variables
        self.n_layers = n_layers
        self.dilation = dilation
        self.kernel_size = kernel_size

        # Define dropout, and activation functions
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Define the network architecture
        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        # Create down-sample branch if input and output channels differ
        self.down_sample = DownSample(in_channels, out_channels, groups) if in_channels != out_channels else None

        for i in range(n_layers):
            self.convolutions.append(nn.Conv1d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                groups=groups
            ))
            self.normalizations.append(nn.GroupNorm(
                num_groups=groups,
                num_channels=out_channels
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Temporal Block to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, new_sequence_length).
        """
        # Set correct dilations
        for conv in self.convolutions:
            conv.dilation = self.dilation

        # Save input tensor to be used as identity
        identity = x

        # If necessary, pass input tensor through down-sample branch
        if self.down_sample is not None:
            identity = self.down_sample(identity)

        for i in range(self.n_layers - 1):
            # Apply the layer
            x = F.pad(x, (self.dilation * (self.kernel_size - 1), 0), 'constant', 0)
            x = self.convolutions[i](x)
            x = self.normalizations[i](x)
            x = self.relu(x)
            x = self.dropout(x)

        # Do not apply relu/dropout to last layer
        x = F.pad(x, (self.dilation * (self.kernel_size - 1), 0), 'constant', 0)
        x = self.convolutions[-1](x)
        x = self.normalizations[-1](x)

        # Add residual connection to the data
        return x + identity
