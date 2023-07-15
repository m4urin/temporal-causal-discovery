import torch
from torch import nn

from src.models.modules.temporal.temporal_module import TemporalModule


class DownSample(TemporalModule):
    """
    A downsampling module for 1D inputs.

    This module first applies a 1D convolution to the input tensor, with groups
    set to the specified number of groups. Then it normalizes the output using
    group normalization.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        groups (int): The number of groups to divide the channels into.

    Attributes:
        conv (nn.Conv1d): A 1D convolutional layer that applies a convolution to
            the input tensor.
        group_norm (nn.GroupNorm): A group normalization layer that normalizes
            the output of the convolutional layer.
    """
    def __init__(self, in_channels: int, out_channels: int, groups: int):
        super().__init__(in_channels, out_channels, groups, receptive_field=1)

        # Initialize the 1D convolutional layer with the specified parameters
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=groups,
            bias=False  # No need for bias, as it will be captured in the group_norm
        )
        # Initialize the group normalization layer with the specified parameters
        self.group_norm = nn.GroupNorm(num_channels=out_channels, num_groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input tensor through the downsampling module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels,
                sequence_length).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, out_channels,
                sequence_length).
        """
        # Apply the convolutional layer to the input tensor, then apply group normalization
        return self.group_norm(self.conv(x))


if __name__ == '__main__':
    d = DownSample(30, 30, 3)
    for n, p in d.named_parameters():
        print(n, p.size())
