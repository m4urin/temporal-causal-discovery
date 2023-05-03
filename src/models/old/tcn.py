

"""
TODO: grouped flag, n_layers within one block
"""
from typing import Tuple

import torch
from torch import nn


class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_layers=1, block_layers=1, groups=1, dropout=0.0):
        """
        A class that implements a Temporal Convolutional Network (TCN).

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The size of the convolution kernel.
            n_layers (int): The number of TCN layers. Default is 1.
            block_layers (int): The number of convolutional layers in each TemporalBlock. Default is 1.
            groups (int): The number of groups to divide the channels into. Default is 1.
            dropout (float): The dropout probability. Default is 0.0.
        """
        super().__init__()
        # Calculate the number of total layers in the network
        self.n_total_layers = n_layers * block_layers

        # Calculate the receptive field of the network
        self.receptive_field = (2 ** n_layers - 1) * block_layers * (kernel_size - 1) + 1

        # Define the network architecture
        self.network = []
        for i in range(n_layers):
            self.network += [TemporalBlock(in_channels=in_channels if i == 0 else out_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           dilation=2 ** i,
                                           groups=groups,
                                           n_layers=block_layers,
                                           dropout=dropout)]
        self.network = nn.Sequential(*self.network)

    def forward(self, x: torch.Tensor):
        # x is of size (..., in_channels, time_steps)
        return self.network(x)


class TemporalConvNetGrouped(nn.Module):
    def __init__(self, num_nodes: int, layer_channels: Tuple[int, ...], kernel_size: int, block_layers: int = 1, dropout: float = 0.0):
        """
        Temporal Convolutional Network (TCN) with grouped convolutions.
        Adjusted from https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks/script

        Transforms a Tensor:
        (batch_size, num_nodes, time_steps)
        -> (batch_size, num_nodes, channels[-1], time_steps)

        Args:
            num_nodes:
                The number of variables / time series (N)
            layer_channels:
                Channels that will be used in the network, e.g. (32, 64) will result in 2 layers
                (!! for every layer, the TCN will create 2 convolution layers with a residual connection)
            kernel_size:
                Kernel size of the convolutions,
                receptive field is (2 * (kernel_size - 1) * (2 ** len(channels) - 1) + 1)
            dropout:
                Dropout probability of units in hidden layers
            """
        super(TemporalConvNetGrouped, self).__init__()
        n_layers = block_layers * len(layer_channels)
        self.receptive_field = block_layers * (kernel_size - 1) * (2 ** n_layers - 1) + 1

        layer_channels = [1] + list(layer_channels)
        temporal_blocks = []
        for i in range(n_layers):
            temporal_blocks += [TemporalBlock(num_nodes=num_nodes, in_channels=layer_channels[i],
                                              out_channels=layer_channels[i + 1], kernel_size=kernel_size,
                                              dilation=2 ** i, dropout=dropout)]
        self.temporal_blocks = nn.Sequential(*temporal_blocks)

        self.init_weights()

    def init_weights(self):
        for temporal_block in self.temporal_blocks:
            temporal_block.init_weights()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of size (batch_size, num_nodes, time_steps)
        :returns: Tensor of size (batch_size, num_nodes, channels[-1], time_steps)
        """
        return self.temporal_blocks(x.unsqueeze(-2))

