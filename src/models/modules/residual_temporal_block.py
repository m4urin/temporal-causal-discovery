import torch
import torch.nn as nn

from src.models.modules.downsample import DownSample


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups, *modules):
        """
        A PyTorch module that implements a series of residual modules with skip connection between modules.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            groups (int): The number of groups.
            *modules (nn.Module): A variable-length list of PyTorch Modules that implement residual modules.
        """
        super().__init__()

        # Store input and output channel counts and residual block modules
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(*modules)

        # Create downsample branch if input and output channels differ
        self.downsample = DownSample(in_channels, out_channels, groups=groups) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualTemporalBlock.

        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            Output tensor of shape (batch_size, out_channels, sequence_length).
        """
        # Save input tensor to be used as identity
        identity = x

        # If necessary, pass input tensor through downsample branch
        if self.downsample is not None:
            identity = self.downsample(identity)

        # Pass input tensor through residual modules
        x = self.block(x)

        # Add identity tensor (after downsample if present) to output of residual modules
        x += identity

        return x
