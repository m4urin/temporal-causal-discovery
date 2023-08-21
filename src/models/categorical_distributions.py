import torch
from torch import nn


class GumbelSoftmax(nn.Module):
    def __init__(self, tau: float = 1, dim=-1):
        """
        Initializes a Gumbel Softmax module.

        Args:
            tau (float): The temperature parameter for controlling the relaxation during sampling.
            dim (int): The dimension along which the softmax is applied.
        """
        super().__init__()
        self.dim = dim
        self.tau_inverse = 1.0 / tau

    def forward(self, x):
        """
        Forward pass of the Gumbel Softmax module. The sampling is only applied during training.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying Gumbel Softmax.
        """
        if self.training:
            # Gumbel-softmax sampling: add Gumbel noise to the input and then apply softmax
            x = x - torch.empty_like(x).exponential_().log()
        return torch.softmax(self.tau_inverse * x, dim=self.dim)


class Softmax_1(nn.Module):
    def __init__(self, dim=-1):
        """
        Initializes a Softmax_1 module.

        Args:
            dim (int): The dimension along which the softmax is applied.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass of the Softmax_1 module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying Softmax_1.
        """
        maxes = torch.max(x, dim=self.dim, keepdim=True)[0]
        x_exp = torch.exp(x - maxes)
        return x_exp / (torch.exp(-maxes) + torch.sum(x_exp, dim=self.dim, keepdim=True))


class NormalizedSigmoid(nn.Module):
    def __init__(self, dim=-1):
        """
        Initializes a NormalizedSigmoid module.

        Args:
            dim (int): The dimension along which the sigmoid is applied.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass of the NormalizedSigmoid module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying NormalizedSigmoid.
        """
        z = torch.sigmoid(x)
        return z / (1e-9 + torch.sum(z, dim=self.dim, keepdim=True))


class SparseMax(nn.Module):
    def __init__(self, dim=-1):
        """
        Initializes a SparseMax module.

        Args:
            dim (int): The dimension along which the SparseMax is applied.
        """
        super(SparseMax, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SparseMax module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying SparseMax.
        """
        # Sort the input tensor in descending order along the specified dimension
        x_sorted = torch.sort(x, dim=self.dim, descending=True).values

        # Compute the cumulative sum along the specified dimension
        x_cumsum = torch.cumsum(x_sorted, dim=self.dim)

        # Create a tensor representing the range of values from 1 to the size of the specified dimension
        k = torch.arange(1, x.size(self.dim) + 1, device=x.device).view(-1, *((1,) * ((-self.dim - 1) % x.dim())))

        # Compute the intermediate array used in sparsemax calculation
        k_array = k * x_sorted + 1

        # Identify the selected elements using the intermediate array
        k_selected = k_array > x_cumsum

        # Compute the maximum index for each element along the specified dimension
        k_max = k_selected.sum(dim=self.dim, keepdim=True)

        # Compute the threshold value for each element along the specified dimension
        threshold = (torch.gather(x_cumsum, dim=self.dim, index=k_max - 1) - 1) / k_max

        # Apply the sparsemax operation by subtracting the threshold and clamping the result to 0.0 or greater
        return torch.clamp(x - threshold, min=0.0)
