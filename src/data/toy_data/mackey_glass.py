from typing import Optional

import torch

from src.data.timeseries_data import TimeSeriesData


def mackey_glass(sequence_length: int, warmup: Optional[int] = 500, reduce: Optional[int] = 6,
                 noise: Optional[float] = 0.1, beta: Optional[float] = 0.2, gamma: Optional[float] = 0.1,
                 tau: Optional[int] = 20, n: Optional[int] = 10) -> TimeSeriesData:
    """
    A function that generates a Mackey-Glass time series dataset

    Args:
        sequence_length: int, length of the sequence
        warmup: int, number of initial values to ignore
        reduce: int, factor by which to reduce the sequence
        noise: float, amount of noise to add to the sequence
        beta: float, the parameter beta in the Mackey-Glass equation
        gamma: float, the parameter gamma in the Mackey-Glass equation
        tau: int, the parameter tau in the Mackey-Glass equation
        n: int, the parameter n in the Mackey-Glass equation

    Returns:
        A TemporalDataset object containing the generated dataset
    """
    # Generate sequence of random values with added noise
    x = torch.randn(tau + warmup + sequence_length * reduce) * noise

    # Generate sequence using the Mackey-Glass equation
    for i in range(tau, len(x) - 1):
        x[i + 1] += (1.0 - gamma) * x[i] + beta * x[i - tau] / (1 + x[i - tau] ** n)

    # Remove warmup period from the sequence
    x = x[tau + warmup:]

    # Reduce the sequence by a factor of reduce
    x = x[torch.arange(0, len(x), reduce, dtype=torch.long)]

    # Add an additional dimension to the tensor to represent a batch of size 1
    x = x.unsqueeze(dim=0).unsqueeze(dim=0)
    print(x.size())

    # Create a TemporalDataset object from the generated sequence
    dataset = TimeSeriesData(name='mackey_glass', train_data=x, normalize=True)

    return dataset
