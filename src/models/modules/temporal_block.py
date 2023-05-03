import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """
    A PyTorch module that implements a Temporal Block.

    This module consists of a sequence of 1D convolutional layers with dilations, followed by ReLU activations
    and dropout layers. The number of layers, kernel size, dilation, and dropout rate can be configured
    during initialization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation (int, optional): Dilation factor for the convolutional layers (default: 1).
        groups (int, optional): Number of groups to split the input and output channels into (default: 1).
        n_layers (int, optional): Number of convolutional layers in the block (default: 1).
        dropout (float, optional): Dropout rate to use between layers (default: 0.0).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        n_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()

        # Check that the number of input and output channels is divisible by the number of groups
        assert (
            in_channels % groups == 0 and out_channels % groups == 0
        ), "Both 'in_channels' and 'out_channels' should be a multiple of 'groups'."

        assert n_layers > 0, "Number of layers should be 1 or greater."

        # Set class variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.receptive_field = n_layers * dilation * (kernel_size - 1) + 1

        # Define padding, dropout, and activation functions
        self.padding = nn.ConstantPad1d((dilation * (kernel_size - 1), 0), 0)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Define the network architecture
        self.network = nn.ModuleList()
        for i in range(n_layers):
            self.network.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        in_channels=in_channels if i == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        groups=groups
                    )
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Temporal Block to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, new_sequence_length).
        """
        output = x

        for i in range(self.n_layers - 1):
            # Apply the layer
            output = self.padding(output)
            output = self.network[i](output)
            output = self.relu(output)
            output = self.dropout(output)

        # Do not apply ReLU to last layer
        output = self.padding(output)
        output = self.network[-1](output)

        return output


def test():
    """
    Test the TemporalBlock module.

    This function creates a TemporalBlock module with some example parameters and applies it to an input tensor.
    It checks that the output shape is as expected.
    """

    # Define example input parameters
    batch_size = 4
    in_channels = 9
    out_channels = 48
    sequence_length = 100
    kernel_size = 3
    dilation = 2
    groups = 3
    n_layers = 2
    dropout = 0.2

    # Create example input tensor
    x = torch.randn(batch_size, in_channels, sequence_length)

    # Create the TemporalBlock module
    tb = TemporalBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        dilation=dilation,
        groups=groups,
        n_layers=n_layers,
        dropout=dropout,
    )
    print(tb)

    # Check that the receptive field is computed correctly
    assert tb.receptive_field == 9

    # Apply the module to the input tensor
    output = tb(x)

    # Check that the output shape is as expected
    assert output.shape == (batch_size, out_channels, sequence_length)

    print("Test passed!")


if __name__ == '__main__':
    test()
