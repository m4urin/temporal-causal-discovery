import math
from typing import Tuple

import torch
import torch.nn as nn


class NAVAR_MASKED(nn.Module):
    def __init__(self, num_nodes: int, kernel_size: int, hidden_dim: int, n_layers: int, dropout: float = 0.0):
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
        super(NAVAR_MASKED, self).__init__()
        self.num_nodes = num_nodes
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_channels=num_nodes * num_nodes,
                                                    out_channels=num_nodes * hidden_dim,
                                                    kernel_size=kernel_size, groups=num_nodes))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(in_channels=num_nodes * hidden_dim,
                                                    out_channels=num_nodes * num_nodes,
                                                    kernel_size=kernel_size, groups=num_nodes, bias=False))
        self.receptive_field = 2 * kernel_size - 1
        pad = nn.ZeroPad2d(((kernel_size - 1), 0, 0, 0))
        self.convolutions = nn.Sequential(pad, self.conv1, nn.ReLU(), nn.Dropout(dropout), pad, self.conv2)

        self.biases = nn.Parameter(torch.empty(num_nodes, 1))
        self.attention = nn.Parameter(torch.empty(num_nodes * num_nodes, 1))

        attention_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool).fill_diagonal_(True)
        data_mask = torch.ones(num_nodes, num_nodes, dtype=torch.bool)
        for i in range(num_nodes):
            data_mask[i, :i + 1] = False

        self.attention_mask = attention_mask.reshape(num_nodes * num_nodes, 1)
        self.data_mask = data_mask.reshape(num_nodes * num_nodes, 1)

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        nn.init.constant_(self.biases, 0.00001)
        nn.init.constant_(self.attention, 0.001)

    def to(self, device, **kwargs):
        self.data_mask = self.data_mask.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return super().to(device=device, **kwargs)

    def forward(self, x: torch.Tensor):
        """
        x: Tensor of size (batch_size, num_nodes, time_steps)
        :returns:
            Tensor 'predictions' (batch_size, num_nodes, time_steps),
            Tensor 'contributions' (batch_size, num_nodes, num_nodes, time_steps),
            Tensor 'attentions' (num_nodes, num_nodes),
        """
        bs = x.shape[0]
        # attn: (num_nodes * num_nodes, 1)
        attention = self.attention * self.data_mask + self.attention_mask

        # x: (batch_size, num_nodes * num_nodes, time_steps)
        x = x.repeat(1, self.num_nodes, 1) * attention

        # x: (bs, num_nodes * num_nodes, time_steps) -> (bs, num_nodes * num_nodes, time_steps)
        x = self.convolutions(x)

        # x: (bs, num_nodes, num_nodes, time_steps)
        contributions = x.view(bs, self.num_nodes, self.num_nodes, -1)

        # predictions: (bs, num_nodes, time_steps)
        predictions = contributions.sum(dim=1) + self.biases

        return predictions, contributions, attention.abs().view(self.num_nodes, self.num_nodes)


if __name__ == '__main__':
    # For testing purposes

    from definitions import DEVICE
    from src.utils import count_params

    model = NAVAR_MASKED(num_nodes=5, kernel_size=3, hidden_dim=32, dropout=0.1).to(DEVICE)
    print(model)
    data = torch.rand((1, 5, 300), device=DEVICE)  # batch_size=1, num_nodes=5, time_steps=300
    predictions_, contributions_, attn_ = model(data)

    print(model)
    print(f"\nn_parameters_per_node={count_params(model) // 5}"
          f"\nreceptive_field={model.receptive_field}"
          f"\npredictions={predictions_.size()}"
          f"\ncontributions={contributions_.size()}"
          f"\nattentions={attn_.size()}")
    print(attn_)