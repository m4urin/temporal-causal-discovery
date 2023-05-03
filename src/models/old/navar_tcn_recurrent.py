import math
from typing import Tuple

import torch
import torch.nn as nn

from src.models.old.navar import NAVAR
from src.utils2 import share_parameters, count_parameters


class TemporalBlockGrouped(nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.0):
        """
        Temporal block with two layers of grouped convolutions with kernel_size and dilation.
        Adjusted from https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks/script

        Transforms a Tensor:
        (batch_size, num_nodes, in_channels, time_steps)
        -> (batch_size, num_nodes, out_channels, time_steps)

        Args:
            num_nodes:
                The number of variables / time series (N)
            in_channels:
                Number of input channels
            out_channels:
                Number of output channels
            kernel_size:
                Kernel size of the convolutions, receptive field is (2 * (kernel_size - 1) * dilation + 1)
            dilation:
                Dilation of the convolutions, receptive field is (2 * (kernel_size - 1) * dilation + 1)
            dropout:
                Dropout probability of units in hidden layers
        """
        super(TemporalBlockGrouped, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.receptive_field = 2 * (kernel_size - 1) * dilation + 1

        self.conv = nn.utils.weight_norm(nn.Conv1d(in_channels=num_nodes * in_channels,
                                                   out_channels=num_nodes * out_channels,
                                                   kernel_size=kernel_size, stride=1,
                                                   dilation=dilation, groups=num_nodes))

        pad = nn.ZeroPad2d(((kernel_size - 1) * dilation, 0, 0, 0))
        dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.convolutions = nn.Sequential(pad, self.conv, self.relu, dropout)

        if in_channels != out_channels:
            self.down_sample = nn.Conv1d(num_nodes * in_channels, num_nodes * out_channels, 1, groups=num_nodes)
        else:
            self.down_sample = None

        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)
        if self.down_sample is not None:
            self.down_sample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor):
        """
        x: Tensor of size (batch_size, num_nodes, in_channels, time_steps)
        :returns: Tensor of size (batch_size, num_nodes, out_channels, time_steps)
        """
        bs = x.shape[0]
        x = x.view(bs, self.num_nodes * self.in_channels, -1)
        residual = x if self.down_sample is None else self.down_sample(x)
        x = self.convolutions(x)
        x = self.relu(x + residual)
        return x.view(bs, self.num_nodes, self.out_channels, -1)


class RecurrentTemporalConvNetGrouped(nn.Module):
    def __init__(self, num_nodes: int, channels: Tuple[int, ...], n_layers: int, kernel_size: int, dropout: float = 0.0):
        """
        Temporal Convolutional Network (TCN) with grouped convolutions.
        Adjusted from https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks/script

        Transforms a Tensor:
        (batch_size, num_nodes, time_steps)
        -> (batch_size, num_nodes, channels[-1], time_steps)

        Args:
            num_nodes:
                The number of variables / time series (N)
            channels:
                Channels that will be used in the network, e.g. (32, 64) will result in 2 layers
                (!! for every layer, the TCN will create 2 convolution layers with a residual connection)
            kernel_size:
                Kernel size of the convolutions,
                receptive field is (2 * (kernel_size - 1) * (2 ** len(channels) - 1) + 1)
            dropout:
                Dropout probability of units in hidden layers
            """
        super(RecurrentTemporalConvNetGrouped, self).__init__()
        assert n_layers >= len(channels)
        self.receptive_field = 2 * (kernel_size - 1) * (2 ** n_layers - 1) + 1

        channels = [1] + list(channels)
        temporal_blocks = []
        for i in range(len(channels)):
            temporal_blocks += [TemporalBlockGrouped(num_nodes=num_nodes, in_channels=channels[i],
                                                     out_channels=channels[i + 1], kernel_size=kernel_size,
                                                     dilation=2 ** i, dropout=dropout)]
        for i in range(len(channels), n_layers):
            temporal_blocks += [TemporalBlockGrouped(num_nodes=num_nodes, in_channels=channels[-1],
                                                     out_channels=channels[-1], kernel_size=kernel_size,
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


class NAVAR_TCN(NAVAR):
    def __init__(self, num_nodes: int, kernel_size: int, n_layers: int, hidden_dim: int, lambda1: float, dropout: float = 0.0):
        """
        Neural Additive Vector AutoRegression (NAVAR) model using a
        Temporal Convolutional Network (TCN) with grouped convolutions.

        Transforms an input Tensor to Tensors 'prediction' and 'contributions':
        (batch_size, num_nodes, time_steps)
        -> (batch_size, num_nodes, time_steps), (batch_size, num_nodes, num_nodes, time_steps)

        Args:
            num_nodes:
                The number of variables / time series (N)
            kernel_size:
                Kernel_size used by the TCN
            n_layers:
                Number of layers used by the TCN (!! for every layer, the tcn creates 2
                convolution layers with a residual connection)
            hidden_dim: int
                Dimensions of the hidden layers
            dropout: float
                Dropout probability of units in hidden layers
        """
        super(NAVAR_TCN, self).__init__(num_nodes, kernel_size, n_layers, hidden_dim, lambda1, dropout)
        self.tcn = RecurrentTemporalConvNetGrouped(num_nodes=num_nodes, channels=(hidden_dim,) * n_layers,
                                                   kernel_size=kernel_size, dropout=dropout)

        # This implements a stacked version of nn.Linear that will be computed in parallel on the GPU
        # with use of torch.einsum
        # The nn.Linear() equivalent: [nn.Linear(hidden_dim, num_nodes, bias=False) for _ in range(num_nodes)]
        self.contributions = nn.Parameter(torch.empty((num_nodes, num_nodes, hidden_dim)))

        self.biases = nn.Parameter(torch.zeros(num_nodes, 1))

        # store in dict, so it will show in the architecture when using print()
        self.output = nn.ParameterDict({'contributions': self.contributions, 'biases': self.biases})

        self.init_weights()

    def init_weights(self):
        self.tcn.init_weights()
        nn.init.kaiming_uniform_(self.contributions, a=math.sqrt(5))
        nn.init.zeros_(self.biases)

    def forward(self, x: torch.Tensor, **kwargs):
        """
        x: Tensor of size (batch_size, num_nodes, time_steps)
        :returns:
            Tensor 'predictions' (batch_size, num_nodes, time_steps),
            Tensor 'contributions' (batch_size, num_nodes, num_nodes, time_steps)
        """
        # x: (bs, num_nodes, time_steps) -> (bs, num_nodes, 1, time_steps) -> (bs, num_nodes, channel_dim, time_steps)
        x = self.tcn(x)

        # contributions: (bs, num_nodes, channel_dim, time_steps) -> (bs, num_nodes, num_nodes, time_steps)
        # b=batch_size, n=num_nodes(number of networks), i=in_features(channel_dim), o=out_features(num_nodes), t=time
        contributions = torch.einsum('noi,bnit->bnot', self.contributions, x)

        # predictions: (bs, num_nodes, time_steps)
        predictions = contributions.sum(dim=1) + self.biases

        return predictions, contributions

    def get_receptive_field(self):
        return self.tcn.receptive_field

    def get_loss(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        #print(x.size(), y.size())
        predictions, contributions = self.forward(x)
        #print(predictions.size())
        regression_loss = ((predictions - y) ** 2).mean()
        regularization_loss = contributions.abs().sum(dim=(1, 2)).mean()
        return regression_loss + (self.lambda1 / self.num_nodes) * regularization_loss

    def evaluate(self, x: torch.Tensor, monte_carlo_dropout: int = None, **kwargs):
        if monte_carlo_dropout is None:
            self.eval()  # disable dropout
            predictions, contributions = self.forward(x)
        else:
            assert monte_carlo_dropout > 1
            self.train()  # enable dropout
            bs, num_nodes, _ = x.shape
            x = x.expand(monte_carlo_dropout, -1, -1)
            predictions, contributions = self.forward(x)
            predictions = predictions.view(monte_carlo_dropout, bs, num_nodes, -1)
            contributions = contributions.view(monte_carlo_dropout, bs, num_nodes, num_nodes, -1)
        predictions = predictions.detach().transpose(-1, -2).cpu()
        contributions = contributions.detach().std(dim=-1).cpu()
        return predictions, contributions


if __name__ == '__main__':
    # For testing purposes

    t0 = TemporalBlockGrouped(num_nodes=3, in_channels=16, out_channels=16, kernel_size=2, dilation=4, dropout=0.1)
    t1 = TemporalBlockGrouped(num_nodes=3, in_channels=16, out_channels=16, kernel_size=2, dilation=8, dropout=0.1)
    for m in [t0, t1]:
        for k, v in m.state_dict().items():
            print(k, v.size())
        print('\n=====================\n')

    print('params t0', count_parameters(t0))
    print('params t1', count_parameters(t1))
    mymodel = nn.Module()
    mymodel.t0 = t0
    mymodel.t1 = t1
    print('params my_model', count_parameters(mymodel))
    print('dilation t0', t0.conv.dilation)
    print('dilation t1', t1.conv.dilation)
    print(t0.conv.bias.data[:10])
    print(t1.conv.bias.data[:10])
    t0.conv.bias.data[0] = 4.0
    print(t0.conv.bias.data[:10])
    print(t1.conv.bias.data[:10])

    print("\n================================\n")

    t0 = TemporalBlockGrouped(num_nodes=3, in_channels=16, out_channels=16, kernel_size=2, dilation=4, dropout=0.1)
    t1 = TemporalBlockGrouped(num_nodes=3, in_channels=16, out_channels=16, kernel_size=2, dilation=8, dropout=0.1)
    share_parameters(t0, t1)

    print('params t0', count_parameters(t0))
    print('params t1', count_parameters(t1))
    mymodel = nn.Module()
    mymodel.t0 = t0
    mymodel.t1 = t1
    print('params my_model', count_parameters(mymodel))
    print('dilation t0', t0.conv.dilation)
    print('dilation t1', t1.conv.dilation)
    print(t0.conv.bias.data[:10])
    print(t1.conv.bias.data[:10])
    t0.conv.bias.data[0] = 4.0
    print(t0.conv.bias.data[:10])
    print(t1.conv.bias.data[:10])
