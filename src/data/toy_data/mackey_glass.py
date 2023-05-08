import torch

from src.data.dataset import normalize, Dataset


def mackey_glass(sequence_length: int, warmup: int = 500, reduce: int = 6, noise: float = 0.1,
                 beta: float = 0.2, gamma: float = 0.1, tau: int = 20, n: int = 10) -> Dataset:
    x = torch.randn(tau + warmup + (sequence_length * reduce)) * noise

    for i in range(tau, len(x) - 1):
        x[i + 1] += (1.0 - gamma) * x[i] + ((beta * x[i - tau]) / (1 + (x[i - tau] ** n)))

    x = x[tau + warmup:]
    x = x[torch.arange(0, len(x), reduce, dtype=torch.long)]
    x = x.unsqueeze(dim=0)
    x = normalize(x, dim=-1)
    return Dataset(x)

