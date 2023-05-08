import torch

from src.data.dataset import normalize, Dataset


def toy_data_3_nodes(batch_size: int = 1, time_steps: int = 500, warmup: int = 200) -> Dataset:
    """
    :return: Dataset object with shape (n_batches, 3_nodes, T), max_lags=1
    """
    assert batch_size > 0 and time_steps > 0 and warmup >= 0
    data = torch.randn((batch_size, 3, warmup + time_steps), dtype=torch.float64)
    for i in range(1, data.size()[-1]):
        data[:, 0, i] += torch.cos(data[:, 1, i - 1]) + torch.tanh(data[:, 2, i - 1])
        data[:, 1, i] += 0.35 * data[:, 1, i - 1] + data[:, 2, i - 1]
        data[:, 2, i] += torch.abs(0.5 * data[:, 0, i - 1]) + torch.sin(2 * data[:, 1, i - 1])
    data = data[..., warmup:]
    data = normalize(data, dim=-1)
    # create Dataset object
    return Dataset(data.to(dtype=torch.float32))

