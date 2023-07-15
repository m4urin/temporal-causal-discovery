import torch
import torch.nn as nn


class TemporalModule(nn.Module):
    """
    A PyTorch module that applies an operation to temporal input data.

    This module expects an input Tensor of size (batch_size, in_channels, sequence_length).
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            groups: int,
            receptive_field = 1
    ) -> None:
        """
        Initializes the TemporalModule.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            groups (int): Number of groups for grouped convolutions.
            receptive_field (Any): Size of the receptive field. Defaults to 1.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.receptive_field = receptive_field
        self.in_dim = in_channels // groups
        self.out_dim = out_channels // groups

    def group_view(self, x: torch.Tensor, groups=None):
        return x.reshape(x.size(0), self.groups if groups is None else groups, -1, x.size(-1))

    def channel_view(self, x: torch.Tensor):
        return x.reshape(x.size(0), -1, x.size(-1))

