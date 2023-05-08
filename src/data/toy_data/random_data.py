import torch

from src.data.dataset import Dataset


def random_timeseries(batch_size: int = 1, num_variables: int = 3, sequence_length: int = 500) -> Dataset:
    """
    :return: Dataset object with shape (n_batches, 3_nodes, T), max_lags=1
    """
    return Dataset(torch.randn((batch_size, num_variables, sequence_length), dtype=torch.float32))
