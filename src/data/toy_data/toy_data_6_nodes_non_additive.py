import torch

from src.data.dataset import normalize, Dataset


def toy_data_6_nodes_non_additive(batch_size: int = 1, time_steps: int = 500, warmup: int = 200):
    """
    :return: Normalized Tensor of size (n_batches, 6_nodes, T), max_lags=8
    """
    assert batch_size > 0 and time_steps > 0 and warmup >= 0
    data = torch.randn((batch_size, 6, warmup + time_steps), dtype=torch.float64) * 0.9
    data[:, 5] *= 0.01
    for i in range(1, data.size()[-1]):
        data[:, 0, i] += 0.5 * torch.sigmoid(data[:, 1, i - 3])
        data[:, 1, i] += 0.2 * data[:, 0, i - 3] * data[:, 4, i - 11]
        data[:, 2, i] += torch.sqrt(0.5 * data[:, 1, i - 1].abs()) * torch.tanh(0.5 * data[:, 4, i - 4])
        data[:, 3, i] = torch.sin(data[:, 2, i - 6]) + 0.05 * data[:, 0, i - 5]
        data[:, 4, i] += torch.tanh(0.5 * data[:, 2, i - 4] * data[:, 3, i - 13])
        # mackey glass
        data[:, 5, i] += 0.8 * data[:, 5, i-1] + (0.1 / (0.3 + (data[:, 5, i-2] ** 2)))

    data = data[..., warmup:]
    data = normalize(data, dim=-1)
    return Dataset(data.to(dtype=torch.float32))
