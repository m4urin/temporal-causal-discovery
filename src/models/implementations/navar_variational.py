import torch
import torch.nn as nn

from src.models.implementations.navar import NAVAR, test_model


class NAVAR_Variational(NAVAR):
    def __init__(self, num_variables: int, kernel_size: int, n_layers: int,
                 hidden_dim: int, lambda1: float, dropout: float = 0.0):
        """
        Neural Additive Vector AutoRegression (NAVAR) model using a
        Variational output layer for Aleatoric uncertainty

        Args:
            num_variables:
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
        super().__init__(num_variables, kernel_size, n_layers, hidden_dim, lambda1, dropout)
        conv = nn.utils.weight_norm(nn.Conv1d(in_channels=num_variables * 1,
                                              out_channels=num_variables * hidden_dim,
                                              kernel_size=kernel_size, groups=num_variables))
        linear = nn.Conv1d(in_channels=num_variables * hidden_dim,
                           out_channels=num_variables * num_variables * 2,
                           kernel_size=1, groups=num_variables)
        self.biases = nn.Parameter(torch.empty(num_variables, 1))
        self.network = nn.Sequential(conv, nn.ReLU(), nn.Dropout(p=dropout), linear)

        conv.weight.data.normal_(0, 0.01)
        linear.weight.data.normal_(0, 0.01)
        self.biases.data.fill_(0.0001)

    def get_receptive_field(self):
        return self.kernel_size

    def get_loss(self, x: torch.Tensor, y: torch.Tensor, sample=True, **kwargs):
        # predictions: (bs, num_nodes, t), mu/std/var: (bs, num_nodes, num_nodes, t)
        predictions, mu, std, log_var = self.forward(x, sample)
        regression_loss = ((predictions - y[..., self.kernel_size - 1:]) ** 2).matrix()
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=(1, 2)))
        return regression_loss + (self.lambda1 / self.num_variables) * kld_loss

    def evaluate(self, x: torch.Tensor, monte_carlo_dropout: int = None, **kwargs):
        if monte_carlo_dropout is None:
            self.eval()  # disable dropout
            predictions, mu, std, _ = self.forward(x, sample=False)
        else:
            assert monte_carlo_dropout > 1
            self.train()  # enable dropout
            bs, num_nodes, _ = x.shape
            x = x.expand(monte_carlo_dropout, -1, -1)
            predictions, mu, std, _ = self.forward(x, sample=False)
            predictions = predictions.view(monte_carlo_dropout, bs, num_nodes, -1)
            mu = mu.view(monte_carlo_dropout, bs, num_nodes, num_nodes, -1)
            std = std.view(monte_carlo_dropout, bs, num_nodes, num_nodes, -1)
        predictions = predictions.detach().transpose(-1, -2).cpu()
        mu = mu.detach().cpu()
        std = std.detach().cpu()
        return predictions, mu, std

    def forward(self, x: torch.Tensor, sample=True, **kwargs):
        """
        x: Tensor of size (batch_size, num_nodes, time_steps)
        :returns:
            Tensor 'predictions' (batch_size, num_nodes, time_steps),
            Tensor 'contributions' (batch_size, num_nodes, num_nodes, time_steps)
            Tensor 'mu' (batch_size, num_nodes, num_nodes, time_steps)
            Tensor 'std' (batch_size, num_nodes, num_nodes, time_steps)
        """
        bs = x.shape[0]

        # x: (bs, num_nodes, time_steps) -> (bs, num_nodes, 2, num_nodes, time_steps)
        x = self.network(x).view(bs, self.num_variables, 2, self.num_variables, -1)

        mu, log_var = x[:, :, 0], x[:, :, 1]

        std = torch.exp(0.5 * log_var)
        if sample:
            eps = torch.randn_like(std)
            contributions = eps * std + mu
        else:
            contributions = mu

        # predictions: (bs, num_nodes, time_steps)
        predictions = contributions.sum(dim=1) + self.biases

        return predictions, mu, std, log_var


if __name__ == '__main__':
    test_model(NAVAR_Variational)
