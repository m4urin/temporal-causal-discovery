import torch
from torch import nn

from src.models.modules.temporal.temporal_module import TemporalModule


class TemporalVariationalLayer(TemporalModule):
    """
    PyTorch module that implements a variational layer using the reparameterization trick.

    This module performs variational inference by taking an input tensor of shape (..., 2 * out_channels),
    where the last dimension represents concatenated vectors of means and log-variances for a set of random
    variables, each of dimension out_channels. It then samples from a normal distribution with these means
    and variances using the reparameterization trick, which allows for end-to-end backpropagation through
    the stochastic operation. The output tensor has shape (batch_size, out_channels, sequence_length).
    """
    def __init__(self, in_channels: int, out_channels: int, groups: int):
        """
        Initialize the variational layer with the given output dimensionality and number of groups.

        Args:
            out_channels (int): The dimensionality of the output.
            groups (int): The number of groups to divide the channels into. Default is 1.
        """
        super().__init__(in_channels, out_channels, groups)
        self.f1 = nn.Conv1d(in_channels, 2 * out_channels, kernel_size=1, groups=groups)

    def forward(self, x: torch.Tensor):
        """
        Perform the forward pass of the variational layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, sequence_length).
        """
        # Reshape the input tensor to (batch_size, groups, in_channels // groups, sequence_length)
        batch_size, _, sequence_length = x.size()

        # (batch_size, 2 * out_channels, seq_len)
        x = self.f1(x)  # (batch_size, 2 * out_channels, seq_len)
        x = self.group_view(x)  # (batch_size, groups, 2 * out_dim, seq_len)

        # Split the input tensor into means and log-variances
        mu, log_var = x.chunk(2, dim=2)  # (batch_size, groups, out_dim, seq_len)

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), dim=(1, 2, 3))  # keep the batch_size

        # Compute standard deviations from log-variances
        std = torch.exp(0.5 * log_var)

        mu = self.channel_view(mu)  # (batch_size, out_channels, seq_len)
        std = self.channel_view(std)  # (batch_size, out_channels, seq_len)

        if self.training:
            # Sample from a normal distribution using the reparameterization trick
            eps = torch.randn_like(std)  # Generate noise from standard normal distribution
            x = mu + eps * std  # Add noise to means and multiply by standard deviations
        else:
            x = mu

        return x, mu, std, kl_loss
