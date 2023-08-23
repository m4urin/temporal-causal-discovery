import math

import torch


def loss_der(y_true, gamma, v, alpha, beta, coeff):
    error = gamma - y_true
    omega = 2.0 * beta * (1.0 + v)

    return torch.mean(
        0.5 * torch.log(math.pi / v)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(error ** 2 * v + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
        + coeff * torch.abs(error) * (2.0 * v + alpha)
    )
