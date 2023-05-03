import torch
import torch.nn as nn
import numpy as np

from matplotlib import pyplot as plt
from torch.optim import AdamW

from definitions import DEVICE
from src.utils2 import iter_with_progress


class NAVAR(nn.Module):
    def __init__(self, num_nodes: int, kernel_size: int, n_layers: int,
                 hidden_dim: int, lambda1: float, dropout: float):
        """
        Neural Additive Vector AutoRegression (NAVAR) model
        Args:
            num_nodes: int
                The number of time series (N)
            hidden_dim: int
                Number of hidden units per layer
            kernel_size: int
                Maximum number of time lags considered (K)
            n_layers: int
                Number of hidden layers
            lambda1: float
                Lambda for regularization
            dropout:
                Dropout probability of units in hidden layers
        """
        super(NAVAR, self).__init__()
        self.num_nodes = num_nodes
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lambda1 = lambda1
        self.dropout = dropout
        self.receptive_field = kernel_size

        padding = nn.ConstantPad1d((kernel_size - 1, 0), 0)
        relu = nn.ReLU()
        dropout = nn.Dropout(dropout)

        layers = []
        for i in range(n_layers):
            conv = nn.utils.weight_norm(
                nn.Conv1d(
                    in_channels=num_nodes if i == 0 else hidden_dim * num_nodes,
                    out_channels=hidden_dim * num_nodes,
                    kernel_size=kernel_size,
                    groups=num_nodes
                )
            )
            layers += [padding, conv, relu, dropout]

        contributions = nn.Conv1d(
            in_channels=hidden_dim * num_nodes,
            out_channels=num_nodes * num_nodes,
            kernel_size=1,
            groups=num_nodes
        )

        self.network = nn.Sequential(*layers, contributions)
        self.biases = nn.Parameter(torch.ones(1, num_nodes, 1) * 0.0001)

    def forward(self, x: torch.Tensor):
        # x is size(batch_size, num_nodes, sequence_length)
        batch_size = x.size(0)

        # contributions is size(batch_size, num_nodes, num_nodes, sequence_length)
        contributions = self.network(x).view(batch_size, self.num_nodes, self.num_nodes, -1)

        # predictions is of size(batch_size, num_nodes, sequence_length)
        predictions = torch.sum(contributions, dim=1) + self.biases

        return predictions, contributions

    def get_loss(self, x):
        # x is size(batch_size, num_nodes, sequence_length)
        batch_size = x.size(0)


def train_model(data, epochs, val_proportion, learning_rate, lambda1,
                dropout, weight_decay, kernel_size, n_layers, hidden_dim,
                show_progress='tqdm', show_plot=False):
    assert show_progress in ['tqdm', 'console']
    # data is of size (batch_size, num_nodes, sequence_length)
    batch_size, num_nodes, sequence_length = data.size()
    n_test = int(val_proportion * sequence_length)
    n_train = sequence_length - n_test
    check_every = 50

    model = NAVAR(num_nodes=num_nodes,
                  kernel_size=kernel_size,
                  n_layers=n_layers,
                  hidden_dim=hidden_dim,
                  lambda1=lambda1,
                  dropout=dropout).to(DEVICE)
    print('receptive field:', model.receptive_field)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_losses, val_losses = [], []

    x_train, y_train = data[..., :n_train - 1], data[..., 1:n_train]
    x_val, y_val = data[..., n_train:-1], data[..., n_train + 1:]

    model.train()
    for i in iter_with_progress(epochs, show_progress):
        # predictions, contributions: (bs, num_nodes, sequence_length), (bs, num_nodes, num_nodes, sequence_length)
        predictions, contributions = model(x_train)
        loss1 = torch.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(loss.item())

        if n_test > 0 and (i % check_every == 0 or i == epochs - 1):
            model.eval()
            with torch.no_grad():
                val_losses.append(model.get_loss(x_val, y_val, sample=False).item())
            model.train()

    if show_plot:
        plt.clf()
        plt.plot(train_losses, label='train loss')
        plt.plot(np.arange(0, len(val_losses)) * check_every, val_losses, label='eval loss')
        plt.show()
        plt.clf()

    with torch.no_grad():
        result = model.evaluate(data, monte_carlo_dropout=20)

    if n_test > 0:
        # training on partial data
        skip_first = 1000 // check_every
        argmin = skip_first + np.argmin(val_losses[skip_first:])
        best_epochs = argmin * check_every
        return model, val_losses[argmin], best_epochs, result
    else:
        # training on complete data
        argmin = np.argmin(train_losses)
        best_epochs = argmin
        return model, train_losses[argmin], best_epochs, result



