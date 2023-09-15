import torch
from torch import nn


class AttentionActivation(nn.Module):
    def forward(self, x, dim=-1):
        """
        Forward pass of the AttentionActivation module.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int): The dimension along which the activation is applied.

        Returns:
            torch.Tensor: Output tensor after applying the AttentionActivation.
        """
        raise NotImplementedError('Not implemented yet.')


class Softmax(AttentionActivation):
    def forward(self, x, dim=-1):
        """
        Forward pass of the Softmax module.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int): The dimension along which the activation is applied.

        Returns:
            torch.Tensor: Output tensor after applying Softmax.
        """
        return torch.softmax(x, dim=dim)


class GumbelSoftmax(AttentionActivation):
    def __init__(self, tau: float = 1):
        """
        Initializes a Gumbel Softmax module.

        Args:
            tau (float): The temperature parameter for controlling the relaxation during sampling.
        """
        super().__init__()
        self.tau_inverse = 1.0 / tau

    def forward(self, x, dim=-1):
        """
        Forward pass of the Gumbel Softmax module. The sampling is only applied during training.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int): The dimension along which the activation is applied.

        Returns:
            torch.Tensor: Output tensor after applying Gumbel Softmax.
        """
        if self.training:
            # Gumbel-softmax sampling: add Gumbel noise to the input and then apply softmax
            x = x - torch.empty_like(x).exponential_().log()
        return torch.softmax(self.tau_inverse * x, dim=dim)


class Softmax_1(AttentionActivation):
    def forward(self, x, dim=-1):
        """
        Forward pass of the Softmax_1 module.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int): The dimension along which the activation is applied.

        Returns:
            torch.Tensor: Output tensor after applying Softmax_1.
        """
        maxes = torch.max(x, dim=dim, keepdim=True)[0]
        x_exp = torch.exp(x - maxes)
        return x_exp / (torch.exp(-maxes) + torch.sum(x_exp, dim=dim, keepdim=True))


class GumbelSoftmax_1(AttentionActivation):
    def __init__(self, tau: float = 1):
        """
        Initializes a Gumbel Softmax module.

        Args:
            tau (float): The temperature parameter for controlling the relaxation during sampling.
        """
        super().__init__()
        self.tau_inverse = 1.0 / tau
        self.softmax1 = Softmax_1()

    def forward(self, x, dim=-1):
        """
        Forward pass of the Gumbel Softmax module. The sampling is only applied during training.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int): The dimension along which the activation is applied.

        Returns:
            torch.Tensor: Output tensor after applying Gumbel Softmax.
        """
        if self.training:
            # Gumbel-softmax sampling: add Gumbel noise to the input and then apply softmax
            x = x - torch.empty_like(x).exponential_().log()
        return self.softmax1(self.tau_inverse * x, dim=dim)


class NormalizedSigmoid(AttentionActivation):
    def forward(self, x, dim=-1):
        """
        Forward pass of the NormalizedSigmoid module.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int): The dimension along which the activation is applied.

        Returns:
            torch.Tensor: Output tensor after applying NormalizedSigmoid.
        """
        z = torch.sigmoid(x)
        return z / (1e-9 + torch.sum(z, dim=dim, keepdim=True))


class SparseMax(AttentionActivation):
    def forward(self, x, dim=-1):
        """
        Forward pass of the SparseMax module.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int): The dimension along which the activation is applied.

        Returns:
            torch.Tensor: Output tensor after applying SparseMax.
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


def get_attention_activation(name: str) -> AttentionActivation:
    """
    Returns a specified softmax activation method as a torch.nn.Module object.

    Args:
        name (str): Name of the desired softmax activation method.

    Raises:
        NotImplementedError: If the provided method name is not valid.
    """
    if name == 'softmax':
        return Softmax()
    elif name == 'softmax-1':
        return Softmax_1()
    elif name == 'normalized-sigmoid':
        return NormalizedSigmoid()
    elif name == 'gumbel-softmax':
        return GumbelSoftmax()
    elif name == 'gumbel-softmax-1':
        return GumbelSoftmax_1()
    elif name == 'sparsemax':
        return SparseMax()
    else:
        raise NotImplementedError(f"Method '{name}' is not a valid method.")
