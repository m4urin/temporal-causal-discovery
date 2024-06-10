import torch
from torch import nn


class GumbelSoftmax(nn.Module):
    def __init__(self, temperature=1.0):
        """
        Initializes the GumbelSoftmax module.

        Args:
        temperature (float): The temperature parameter for the Gumbel-Softmax distribution.
                             A lower temperature makes the output closer to one-hot encoded,
                             and as the temperature goes to infinity, the output approaches
                             a uniform distribution.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, logits, dim):
        """
        Forward pass for the GumbelSoftmax module.

        Args:
        logits (Tensor): Unnormalized log probabilities of shape (batch_size, ...).

        Returns:
        Tensor: Sampled labels.
        """
        if self.training:
            uniform_samples = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(uniform_samples + 1e-20) + 1e-20)
            return torch.softmax((logits + gumbel_noise) / self.temperature, dim=dim)
        else:
            return torch.softmax(logits, dim=dim)


class SoftmaxModule(nn.Module):
    def forward(self, logits, dim):
        """
        Forward pass of the SoftmaxModule.

        Args:
        logits (Tensor): The input tensor containing raw scores to be passed through softmax.

        Returns:
        Tensor: The softmax-normalized probabilities across the specified dimension.
        """
        return torch.softmax(logits, dim=dim)

