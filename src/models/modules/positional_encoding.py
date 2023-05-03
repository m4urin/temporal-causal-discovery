import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Class for computing the positional encoding for transformer input sequences.

    Args:
        channels (int): Number of channels in the input tensor.
        max_len (int, optional): Maximum length of input sequences. Default is 1000.
    """
    def __init__(self, channels: int, max_len: int = 1000) -> None:
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * (-math.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d_model, seq_len).

        Returns:
            torch.Tensor: Tensor with positional encoding added, of shape (batch_size, d_model, seq_len).
        """
        # Add the positional encoding to the input tensor
        x = x + self.pe[:, :, :x.size(-1)]
        return x

    def __str__(self):
        return f"PositionalEncoding(channels={self.pe.shape[1]}, max_len={self.pe.shape[2]})"

    def __repr__(self):
        return str(self)


def test():
    """
    Test the PositionalEncoding module by adding positional encoding to a random
    input tensor and visualizing the positional encodings.
    """
    from matplotlib import pyplot as plt

    # Define the parameters for the test
    batch_size = 2
    channels = 32
    seq_length = 50

    # Create a PositionalEncoding instance
    pe = PositionalEncoding(channels=channels, max_len=2 * seq_length)
    print(pe)

    # Create a sample input tensor of shape (batch_size, channels, seq_length)
    x = torch.randn(batch_size, channels, seq_length)

    # Add positional encoding to the input tensor
    x = pe(x)

    # Verify that the output tensor has the expected shape
    assert x.shape == (batch_size, channels, seq_length)

    # Visualize the positional encodings
    plt.figure(figsize=(16, 6))
    plt.plot(pe.pe[0].t())
    plt.legend([f'channel {i}' for i in range(channels)])
    plt.title('Positional Encoding')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.show()

    print("Test passed!")


if __name__ == '__main__':
    test()
