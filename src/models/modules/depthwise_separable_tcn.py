import torch
import torch.nn as nn
from src.models.modules.residual_temporal_block import ResidualTemporalBlock
from src.models.modules.temporal_block import TemporalBlock


class DepthwiseSeparableTCN(nn.Module):
    """
    Implementation of a Temporal Convolutional Network (TCN) that is depthwise separable.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        n_layers (int, optional): Number of layers in the network. Default is 1.
        block_layers (int, optional): Number of residual modules per layer. Default is 1.
        groups (int, optional): Number of groups in depthwise separable convolution. Default is 1.
        dropout (float, optional): Dropout rate. Default is 0.0.

    Attributes:
        n_total_layers (int): Total number of layers in the network
        receptive_field (int): Receptive field of the network
        network (nn.ModuleList): List of residual modules

    """

    def __init__(self, in_channels, out_channels, kernel_size, n_layers=1, block_layers=1, groups=1, dropout=0.0):
        super().__init__()

        # Set the number of layers, total layers, and receptive field of the network
        self.n_layers = n_layers
        self.n_total_layers = n_layers * block_layers
        self.receptive_field = (2 ** n_layers - 1) * block_layers * (kernel_size - 1) + 1

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Create the network as a list of residual modules
        self.network = nn.ModuleList()
        for i in range(n_layers):
            # Create a temporal block with specified parameters
            temp_block = TemporalBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=2 ** i,
                groups=groups,
                n_layers=block_layers,
                dropout=dropout
            )
            # Create a residual block with the temporal block and add to the network
            residual = ResidualTemporalBlock(temp_block.in_channels, out_channels, temp_block)
            self.network.append(residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the Temporal Convolutional Network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, sequence_length)
        """

        # Pass input tensor through all layers of the network
        out = x
        for i in range(self.n_layers - 1):
            out = self.network[i](out)
            out = self.relu(out)
            out = self.dropout(out)
        # Do not apply ReLU to last layer
        out = self.network[-1](out)
        return out


def test():
    """
    Test the functionality of the DepthwiseSeparableTCN module.

    This function creates an instance of the DepthwiseSeparableTCN module with specified parameters and generates a random input
    tensor with the specified dimensions. It performs a forward pass through the module to generate an output tensor and checks
    that the shape of the output tensor is correct. It also checks that the receptive field and number of layers of the module
    are as expected. Finally, it checks that all layers have the correct number of input and output channels.
    """
    # Define input tensor dimensions
    batch_size = 16
    groups = 3
    in_channels = 3
    out_channels = 48
    sequence_length = 100

    # Create an instance of the DepthwiseSeparableTCN module
    tcn = DepthwiseSeparableTCN(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=3, groups=groups, n_layers=2, block_layers=2)
    print(tcn)

    # Generate a random input tensor
    x = torch.randn(batch_size, in_channels, sequence_length)

    # Perform a forward pass through the module
    out = tcn(x)

    # Check the shape of the output tensor
    assert out.shape == (batch_size, out_channels, sequence_length)

    # Check that the receptive field is correct
    assert tcn.receptive_field == 13

    # Check that the number of layers is correct
    assert tcn.n_total_layers == 4

    # Check that all layers have the correct number of channels
    for i, layer in enumerate(tcn.network):
        assert layer.block[0].in_channels == in_channels if i == 0 else out_channels
        assert layer.block[0].out_channels == out_channels

    print("Test passed!")


if __name__ == '__main__':
    test()
