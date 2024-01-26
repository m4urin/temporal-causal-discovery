import torch


def sliding_window_std(x: torch.Tensor, window: tuple, dim: int = -1, correction: int = 1) -> torch.Tensor:
    """
    Computes the sliding window standard deviation of a tensor along a specified dimension.

    Parameters:
        x (torch.Tensor): Input tensor.
        window (tuple of int): The sliding window (left-inclusive, right). The sum of the tuple is the total window size.
        dim (int, optional): The dimension along which to compute the sliding window standard deviation. Default is the last dimension.
        correction (int, optional): A correction factor for the biased variance. Default is 1.

    Returns:
        torch.Tensor: Tensor of sliding window standard deviations.
    """
    tensor_size = x.size(dim)
    window_size = sum(window)

    x = pad(x, window, dim=dim)  # Assumes pad function is imported and defined elsewhere

    # Cumulative sum of x and x^2
    x_cumsum = torch.cumsum(x, dim=dim)
    x2_cumsum = torch.cumsum(x ** 2, dim=dim)

    # Sum of x and x^2 over the window
    x_sum = x_cumsum.narrow(dim, window_size, tensor_size) - x_cumsum.narrow(dim, 0, tensor_size)
    x2_sum = x2_cumsum.narrow(dim, window_size, tensor_size) - x2_cumsum.narrow(dim, 0, tensor_size)

    # Mean of x and x^2 over the window
    x_mean = x_sum.div(window_size)
    x2_mean = x2_sum.div(window_size)

    # Variance and standard deviation
    variance = x2_mean - x_mean ** 2

    if correction == 1:
        variance = window_size * variance / (window_size - correction)

    std_dev = torch.sqrt(variance)

    return std_dev


def pad(x: torch.Tensor, padding: tuple, dim: int = -1) -> torch.Tensor:
    """
    Pads a given tensor along a specified dimension with the provided padding values.
    padding is a tuple containing the padding values for the left and right sides.
    """
    num_dims = x.dim()
    pad_sizes = [0] * 2 * num_dims
    index = 2 * ((-dim - 1) % num_dims)
    pad_sizes[index] = padding[0]  # Padding to the left
    pad_sizes[index + 1] = padding[1]  # Padding to the right
    return torch.nn.functional.pad(x, pad_sizes)


def weighted_sliding_window_std(x: torch.Tensor, weights: torch.Tensor,
                                window: tuple, dim: int = -1, correction: int = 1) -> torch.Tensor:
    """
    Computes the sliding window weighted standard deviation of a tensor along a specified dimension.

    Parameters:
        x (torch.Tensor): Input tensor.
        weights (torch.Tensor): Weights tensor with the same shape as x.
        window (tuple of int): The sliding window (left-inclusive, right). The sum of the tuple is the total window size.
        dim (int, optional): The dimension along which to compute the sliding window standard deviation. Default is the last dimension.
        correction (int, optional): A correction factor for the biased variance. Default is 1.

    Returns:
        torch.Tensor: Tensor of sliding window weighted standard deviations.
    """
    assert x.shape == weights.shape, "x and weights must have the same shape"

    tensor_size = x.size(dim)
    window_size = sum(window)

    x = pad(x, window, dim=dim)  # Assumes pad function is imported and defined elsewhere
    weights = pad(weights, window, dim=dim)

    # Weighted cumulative sum of x, x^2, and weights
    w_cumsum = torch.cumsum(weights, dim=dim)
    xw_cumsum = torch.cumsum(x * weights, dim=dim)
    x2w_cumsum = torch.cumsum((x ** 2) * weights, dim=dim)

    # Sum of weighted x, x^2, and weights over the window
    w_sum = w_cumsum.narrow(dim, window_size, tensor_size) - w_cumsum.narrow(dim, 0, tensor_size)
    xw_sum = xw_cumsum.narrow(dim, window_size, tensor_size) - xw_cumsum.narrow(dim, 0, tensor_size)
    x2w_sum = x2w_cumsum.narrow(dim, window_size, tensor_size) - x2w_cumsum.narrow(dim, 0, tensor_size)

    # Weighted mean of x, x^2, and weights over the window
    xw_mean = xw_sum.div(w_sum)
    x2w_mean = x2w_sum.div(w_sum)

    # Weighted variance and standard deviation
    variance = x2w_mean - xw_mean ** 2

    if correction == 1:
        w2_cumsum = torch.cumsum(weights.pow(2), dim=dim)
        w2_sum = w2_cumsum.narrow(dim, window_size, tensor_size) - w2_cumsum.narrow(dim, 0, tensor_size)
        variance = variance * w_sum / ((w_sum ** 2) - w2_sum)

    std_dev = torch.sqrt(variance)

    return std_dev

