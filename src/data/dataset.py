import os
import zipfile
import numpy as np
import torch

from definitions import DEVICE, DATA_DIR


class Dataset:
    def __init__(self, data: torch.Tensor, name: str = None):
        self.data = data
        self.batch_size, self.num_variables, self.sequence_length = data.shape
        self.name = name

    def __str__(self):
        return f"Dataset(name={self.name + ', ' if self.name is not None else ''}batch_size={self.batch_size}, " \
               f"num_variables={self.num_variables}, sequence_length={self.sequence_length})"

    def __repr__(self):
        return str(self)

    def cuda(self):
        self.data = self.data.cuda()
        return self

    def cpu(self):
        self.data = self.data.detach().cpu()
        return self

    def __getitem__(self, index):
        data = self.data[index]
        if len(data.shape) == 3:
            return Dataset(data)
        return data


def normalize(x: torch.Tensor, dim: int):
    return (x - x.mean(dim=dim, keepdim=True)) / x.std(dim=dim, keepdim=True)


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


def get_causeme_data(experiment: str) -> Dataset:
    """
    :return: Dataset object with shape (n_datasets, n_batches, n_nodes, T)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{experiment.split('.zip')[0]}.zip")
    with zipfile.ZipFile(file_path, "r") as f:
        # array with (n_exp, T, n_var)
        data = np.stack([np.loadtxt(f.open(name)) for name in sorted(f.namelist())])
    # (n_exp, T, n_var)
    data = torch.from_numpy(data).to(device=DEVICE)
    # standardize (n_exp, T, n_var)
    data = normalize(data, dim=1)
    # create Dataset object
    return Dataset(data.transpose(-1, -2).unsqueeze(dim=1).to(dtype=torch.float32))


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
    return Dataset(data.to(device=DEVICE, dtype=torch.float32))


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
    return Dataset(data.to(device=DEVICE, dtype=torch.float32))


"""
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
    contributions = contributions.abs().std(dim=-1)

    data = data[..., warmup:]
    return Dataset(data.to(dtype=torch.float32), contributions.to(dtype=torch.float32))
"""


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
    return Dataset(data.to(device=DEVICE, dtype=torch.float32))





