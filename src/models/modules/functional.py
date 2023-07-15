import math

import torch


def sparsemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes the sparsemax operation along a specified dimension of the input tensor.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int, optional): The dimension along which to compute sparsemax.
            Defaults to -1.

    Returns:
        torch.Tensor: The output tensor after applying sparsemax.

    Raises:
        IndexError: If the specified dimension is out of range.
    """

    # Sort the input tensor in descending order along the specified dimension
    x_sorted = torch.sort(x, dim=dim, descending=True).values

    # Compute the cumulative sum along the specified dimension
    x_cumsum = torch.cumsum(x_sorted, dim=dim)

    # Create a tensor representing the range of values from 1 to the size of the specified dimension
    k = torch.arange(1, x.size(dim) + 1, device=x.device).view(-1, *((1,) * ((-dim - 1) % x.dim())))

    # Compute the intermediate array used in sparsemax calculation
    k_array = k * x_sorted + 1

    # Identify the selected elements using the intermediate array
    k_selected = k_array > x_cumsum

    # Compute the maximum index for each element along the specified dimension
    k_max = k_selected.sum(dim=dim, keepdim=True)

    # Compute the threshold value for each element along the specified dimension
    threshold = (torch.gather(x_cumsum, dim=dim, index=k_max - 1) - 1) / k_max

    # Apply the sparsemax operation by subtracting the threshold and clamping the result to 0.0 or greater
    return torch.clamp(x - threshold, min=0.0)


def entropy(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    Computes the entropy of a probability distribution along a specified dimension of the input tensor.

    Args:
        x (torch.Tensor): The input tensor representing a probability distribution.
                          Each value should be greater or equal to 0.
        dim (int, optional): The dimension along which to compute the entropy.
            Defaults to -1.
        keepdim (bool, optional): Whether to keep the dimension of the output tensor.
            Defaults to False.

    Returns:
        torch.Tensor: The entropy tensor.

    Raises:
        IndexError: If the specified dimension is out of range.
    """

    # Compute the negative entropy by multiplying the input tensor with its logarithm
    # and dividing it by the logarithm of the tensor size along the specified dimension
    # Adds a small constant to the input tensor to avoid taking the logarithm of zero
    entropy_tensor = -x * torch.log(x + 1e-30) / math.log(x.size(dim))

    # Sum the entropy values along the specified dimension
    return entropy_tensor.sum(dim=dim, keepdim=keepdim)


if __name__ == '__main__':

    #attn = torch.tensor([[0.8, 0.6, 1.0, 0.9], [0.1, 0.1, 0.2, 0.2], [0.0, 0.0, 0.1, 0.1],
    #                     [9, 9, 7, 90], [-6, -1.5, -1, -3], [0, 0, 0, 0],], requires_grad=True)
    #print(attn)
    #attn = sparsemax(attn, dim=-1)
    attn = torch.tensor([[0, 0, 0, 0], [1.0, 0, 0, 0], [0.7, 0.3, 0, 0]])
    print(attn)
    attn = entropy(attn, dim=-1)
    print(attn)
