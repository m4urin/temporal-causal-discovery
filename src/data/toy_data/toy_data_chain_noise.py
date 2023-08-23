import torch

from src.data.timeseries_data import TimeSeriesData


def toy_data_chain_noise(noise: float, batch_size: int = 1, time_steps: int = 500, warmup: int = 200):
    n_variables = 4
    data = torch.randn((batch_size, n_variables, warmup + time_steps))
    data[:, :-1, 1:] *= noise  # noise factor
    data[:, -1] *= 0.01
    contributions = torch.zeros((batch_size, n_variables, n_variables, warmup + time_steps))
    for i in range(1, warmup + time_steps):
        contributions[:, 1, 0, i] = (data[:, 1, i - 1] - 0.3).clamp(min=-0.3, max=0.3)
        contributions[:, 2, 1, i] = (-2 * data[:, 2, i - 1] - 0.2).cos()
        contributions[:, 3, 2, i] = data[:, 3, i - 1].abs()
        contributions[:, 3, 3, i] = torch.cos(torch.tensor(i)/3) + torch.sin(torch.tensor(i) / 10)

        data[:, 0, i] += contributions[:, 1, 0, i]
        data[:, 1, i] += contributions[:, 2, 1, i]
        data[:, 2, i] += contributions[:, 3, 2, i]
        data[:, 3, i] += contributions[:, 3, 3, i]

    mu = data.mean(dim=-1, keepdim=True)
    var = data.std(dim=-1, keepdim=True)
    data -= mu
    data /= var
    contributions -= mu.unsqueeze(dim=1)
    contributions /= var.unsqueeze(dim=1)
    contributions = contributions.abs().std_pred(dim=-1)

    data = data[..., warmup:]
    return TimeSeriesData(name="", train_data=data)