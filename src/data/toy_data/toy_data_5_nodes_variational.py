import torch

from src.data.dataset import normalize, Dataset


def toy_data_5_nodes_variational(batch_size: int = 1, time_steps: int = 500, warmup: int = 200):
    """
    :return: Normalized Tensor of size (n_batches, 6_nodes, T), max_lags=8
    """
    assert batch_size > 0 and time_steps > 0 and warmup >= 0
    data = torch.randn((batch_size, 5, warmup + time_steps), dtype=torch.float64) * 0.9
    data[:, 4] *= 0.01
    for i in range(1, data.size()[-1]):
        data[:, 0, i] += 0.5 * torch.sigmoid(data[:, 3, i - 2])
        data[:, 1, i] += torch.sin(data[:, 0, i - 1])
        data[:, 2, i] += torch.cos(data[:, 1, i - 3].abs())
        factor = min(0.8, ((time_steps + warmup - i) / time_steps))
        data[:, 3, i] = 0.3 * data[:, 0, i - 1] + factor * data[:, 1, i - 1]
        # mackey glass
        data[:, 4, i] += 0.8 * data[:, 4, i - 1] + (0.1 / (0.3 + (data[:, 4, i - 2] ** 2)))

    data = data[..., warmup:]
    data = normalize(data, dim=-1)
    return Dataset(data.to(dtype=torch.float32))
