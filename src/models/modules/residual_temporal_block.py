import torch
import torch.nn as nn


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *modules):
        """
        A PyTorch module that implements a series of residual modules with skip connection between modules.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            *modules (nn.Module): A variable-length list of PyTorch Modules that implement residual modules.
        """
        super().__init__()

        # Store input and output channel counts and residual block modules
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(*modules)

        # Create downsample branch if input and output channels differ
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualTemporalBlock.

        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            Output tensor of shape (batch_size, out_channels, sequence_length).
        """
        # Save input tensor to be used as identity
        identity = x

        # If necessary, pass input tensor through downsample branch
        if self.downsample is not None:
            identity = self.downsample(identity)

        # Pass input tensor through residual modules
        x = self.block(x)

        # Add identity tensor (after downsample if present) to output of residual modules
        x += identity

        return x


def test():
    """
    Tests the functionality of the ResidualTemporalBlock class.
    """
    # Create a ResidualTemporalBlock with 3 residual modules
    rtb = ResidualTemporalBlock(10, 20,
                                nn.Conv1d(10, 20, kernel_size=3, padding=1),
                                nn.BatchNorm1d(20),
                                nn.ReLU(),
                                nn.Conv1d(20, 20, kernel_size=3, padding=1),
                                nn.BatchNorm1d(20),
                                nn.ReLU(),
                                nn.Conv1d(20, 20, kernel_size=3, padding=1),
                                nn.BatchNorm1d(20),
                                nn.ReLU())
    print(rtb)

    # Create a random input tensor
    x = torch.randn(32, 10, 100)

    # Compute the output of the ResidualTemporalBlock
    y = rtb(x)

    # Check that the output has the correct shape
    assert y.shape == (32, 20, 100)

    print("Test passed!")


if __name__ == '__main__':
    test()
