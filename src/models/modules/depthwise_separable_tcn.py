import torch
import torch.nn as nn
from src.models.modules.residual_temporal_block import ResidualTemporalBlock
from src.models.modules.temporal_block import TemporalBlock


class DepthwiseSeparableTCN(nn.Module):
    """
    Implementation of a Temporal Convolutional Network (TCN) that is depthwise separable.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        n_blocks (int, optional): Number of residual blocks in the network. Default is 1.
        n_layers_per_block (int, optional): Number of layers per residual block. Default is 1.
        groups (int, optional): Number of groups in depthwise separable convolution. Default is 1.
        dropout (float, optional): Dropout rate. Default is 0.0.

    Attributes:
        n_total_layers (int): Total number of layers in the network
        receptive_field (int): Receptive field of the model
        network (nn.ModuleList): List of residual modules
    """
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, n_layers_per_block=1, groups=1, dropout=0.0):
        super().__init__()

        # Set the number of layers, total layers, and receptive field of the network
        self.n_blocks = n_blocks
        self.n_total_layers = n_blocks * n_layers_per_block
        self.receptive_field = (2 ** n_blocks - 1) * n_layers_per_block * (kernel_size - 1) + 1

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Create the network as a list of residual modules
        self.network = nn.ModuleList()
        for i in range(n_blocks):
            # Create a temporal block with specified parameters
            temp_block = TemporalBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=2 ** i,
                groups=groups,
                n_layers=n_layers_per_block,
                dropout=dropout
            )
            # Create a residual block with the temporal block and add to the network
            residual = ResidualTemporalBlock(temp_block.in_channels, out_channels, groups, temp_block)
            self.network.append(residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the Temporal Convolutional Network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, sequence_length)
        """

        # Pass input tensor through all layers of the network
        out = x
        for i in range(self.n_blocks):
            out = self.network[i](out)
            out = self.relu(out)
            out = self.dropout(out)
        return out
