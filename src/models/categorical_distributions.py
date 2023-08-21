import torch
from torch import nn


class GumbelSoftmax(nn.Module):
    def __init__(self, tau: float = 1, dim=-1):
        super().__init__()
        self.dim = dim
        self.tau_inverse = 1.0 / tau

    def forward(self, x):
        if self.training:
            x = x - torch.empty_like(x).exponential_().log()
        return torch.softmax(self.tau_inverse * x, dim=self.dim)


class Softmax_1(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = torch.max(x, dim=self.dim, keepdim=True)[0]
        x_exp = torch.exp(x - maxes)
        return x_exp / (torch.exp(-maxes) + torch.sum(x_exp, dim=self.dim, keepdim=True))


class NormalizedSigmoid(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        z = torch.sigmoid(x)
        return z / (1e-9 + torch.sum(z, dim=self.dim, keepdim=True))


class SparseMax(nn.Module):
    def __init__(self, dim=-1):
        super(SparseMax, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the sparsemax operation along a specified dimension of the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying sparsemax.
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
