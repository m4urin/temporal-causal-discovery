import torch
from torch import nn


class DownSample(nn.Module):
    """
    This class implements a downsampling module for 1D inputs.

    The input tensor is first passed through a 1D convolution layer, with groups
    set to the specified number of groups. The output of this layer is then
    normalized using batch normalization. Finally, the tensor is reshaped to its
    original batch size and sequence length, with the number of channels reduced
    by the specified factor of groups.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        groups (int): The number of groups to divide the channels into.

    Attributes:
        group_channels (int): The number of channels in each group.
        conv (nn.Conv1d): The 1D convolution layer.
        batch_norm (nn.BatchNorm1d): The batch normalization layer.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()

        assert out_channels % groups == 0, "out_channels must be divisible by groups"

        self.group_channels = out_channels // groups
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            groups=groups
        )
        self.batch_norm = nn.BatchNorm1d(self.group_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input tensor through the downsampling module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, out_channels, sequence_length).
        """

        # x becomes size (batch_size, out_channels, sequence_length)
        x = self.conv(x)

        # x becomes size (batch_size * groups, out_channels // groups, sequence_length)
        batch_size, out_channels, sequence_length = x.shape
        x = x.view(-1, self.group_channels, sequence_length)

        # x is size (batch_size * groups, out_channels // groups, sequence_length)
        x = self.batch_norm(x)

        # x becomes size (batch_size, out_channels, sequence_length)
        x = x.view(batch_size, out_channels, sequence_length)

        return x
