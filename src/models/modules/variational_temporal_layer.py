import torch
from torch import nn


class VariationalTemporalLayer(nn.Module):
    """
    PyTorch module that implements a variational layer using the reparameterization trick.

    This module performs variational inference by taking an input tensor of shape (..., 2 * out_channels),
    where the last dimension represents concatenated vectors of means and log-variances for a set of random
    variables, each of dimension out_channels. It then samples from a normal distribution with these means
    and variances using the reparameterization trick, which allows for end-to-end backpropagation through
    the stochastic operation. The output tensor has shape (batch_size, out_channels, sequence_length).
    """
    def __init__(self, out_channels: int, groups: int = 1) -> None:
        """
        Initialize the variational layer with the given output dimensionality and number of groups.

        Args:
            out_channels (int): The dimensionality of the output.
            groups (int): The number of groups to divide the channels into. Default is 1.
        """
        super().__init__()
        assert out_channels % groups == 0, "'out_channels' should be a multiple of 'groups'."
        self.out_channels = out_channels
        self.groups = groups
        self.out_channels_per_group = self.out_channels // self.groups

    def forward(self, x: torch.Tensor):
        """
        Perform the forward pass of the variational layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2 * out_channels, sequence_length).
            sample (bool): Flag indicating whether to sample from a normal distribution or not.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, sequence_length).
        """
        # Reshape the input tensor to (batch_size, groups, in_channels // groups, sequence_length)
        batch_size, in_channels, sequence_length = x.shape
        x = x.view(batch_size, self.groups, 2, self.out_channels_per_group, sequence_length)

        # Split the input tensor into means and log-variances
        mu, log_var = x[:, :, 0], x[:, :, 1]

        # Compute standard deviations from log-variances
        std = torch.exp(0.5 * log_var)

        if self.training:
            # Sample from a normal distribution using the reparameterization trick
            eps = torch.randn_like(std)  # Generate noise from standard normal distribution
            x = mu + eps * std  # Add noise to means and multiply by standard deviations
        else:
            x = mu

        # Reshape the output tensor to (batch_size, out_channels, sequence_length)
        output = x.reshape(batch_size, self.out_channels, sequence_length)
        mu = mu.reshape(batch_size, self.out_channels, sequence_length)
        std = std.reshape(batch_size, self.out_channels, sequence_length)
        return output, mu, std

    def __str__(self):
        return f"VariationalTemporalLayer(in_channels={2 * self.out_channels}, out_channels={self.out_channels}, groups={self.groups})"

    def __repr__(self):
        return str(self)


def test():
    """
    Function to test the VariationalTemporalLayer module.

    This function creates an instance of the VariationalTemporalLayer module and performs a forward pass
    with random input. It then checks that the output tensor has the correct shape, and that the means and
    standard deviations have been returned correctly.
    """
    # Set up the input tensor
    batch_size = 8
    channels = 32
    groups = 2
    sequence_length = 100
    input_tensor = torch.randn(batch_size, 2 * channels, sequence_length)

    # Create an instance of the VariationalTemporalLayer module
    vtl = VariationalTemporalLayer(out_channels=channels, groups=groups)
    print(vtl)

    # Perform a forward pass with the input tensor
    output_tensor, mu, std = vtl(input_tensor, sample=True)

    # Check that the output tensor has the correct shape
    assert output_tensor.shape == (batch_size, channels, sequence_length)

    # Check that the means and standard deviations have been returned correctly
    assert mu.shape == (batch_size, channels, sequence_length)
    assert std.shape == (batch_size, channels, sequence_length)

    print("Test passed!")


if __name__ == '__main__':
    test()
