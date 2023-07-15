import torch

from src.data.timeseries_data import normalize, TimeSeriesData


def toy_data_coupled(batch_size: int = 1, time_steps: int = 500, warmup: int = 200):
    """
    :return: Normalized Tensor of size (n_batches, 6_nodes, T), max_lags=8
    """
    assert batch_size > 0 and time_steps > 0 and warmup >= 0
    data = torch.randn((batch_size, 4, warmup + time_steps), dtype=torch.float64) * 0.7
    for i in range(1, data.size()[-1]):
        data[:, 0, i] += 0.5 * torch.sigmoid(data[:, 1, i - 1])
        data[:, 1, i] += data[:, 0, i - 2] * data[:, 2, i - 1]
        data[:, 2, i] += data[:, 0, i - 1].abs() * torch.tanh(0.5 * data[:, 3, i - 2])
        data[:, 3, i] += 0.8 * data[:, 3, i-1] + (0.1 / (0.3 + (data[:, 3, i-2] ** 2)))

    data = data[..., warmup:]
    data = normalize(data, dim=-1)
    return TimeSeriesData(data.to(dtype=torch.float32))
