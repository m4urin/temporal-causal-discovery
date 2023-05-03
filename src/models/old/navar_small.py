import torch
import torch.nn as nn

from src.models.old.navar import NAVAR


class NAVAR_SMALL(NAVAR):
    def __init__(self, num_nodes: int, kernel_size: int, n_layers: int,
                 hidden_dim: int, lambda1: float, dropout: float = 0.0):
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
        super().__init__(num_nodes, kernel_size, n_layers, hidden_dim, lambda1, dropout)
        conv = nn.utils.weight_norm(nn.Conv1d(in_channels=num_nodes * 1,
                                              out_channels=num_nodes * hidden_dim,
                                              kernel_size=kernel_size, groups=num_nodes))
        linear = nn.Conv1d(in_channels=num_nodes * hidden_dim,
                           out_channels=num_nodes * num_nodes,
                           kernel_size=1, groups=num_nodes)
        self.network = nn.Sequential(conv, nn.ReLU(), nn.Dropout(p=dropout), linear)
        self.biases = nn.Parameter(torch.empty(num_nodes, 1))

        conv.weight.data.normal_(0, 0.01)
        linear.weight.data.normal_(0, 0.01)
        self.biases.data.fill_(0.0001)

    def get_receptive_field(self):
        return self.kernel_size

    def get_loss(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        predictions, contributions = self.forward(x)
        regression_loss = ((predictions - y[..., self.kernel_size-1:]) ** 2).mean()
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

    def forward(self, x: torch.Tensor, **kwargs):
        """
        x: Tensor of size (batch_size, num_nodes, time_steps)
        :returns:
            Tensor 'predictions' (batch_size, num_nodes, time_steps),
            Tensor 'contributions' (batch_size, num_nodes, num_nodes, time_steps)
        """
        bs = x.shape[0]

        # x: (bs, num_nodes, time_steps) -> (bs, num_nodes, num_nodes, time_steps)
        contributions = self.network(x).view(bs, self.num_nodes, self.num_nodes, -1)

        # predictions: (bs, num_nodes, time_steps)
        predictions = contributions.sum(dim=1) + self.biases

        return predictions, contributions


if __name__ == '__main__':
    from src.models.old.navar import test_model
    test_model(NAVAR_SMALL)
