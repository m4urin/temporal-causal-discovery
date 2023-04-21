import os
import zipfile
import numpy as np
import torch
from matplotlib import pyplot as plt

from definitions import DEVICE, DATA_DIR
from src.utils import join_path


def normalize(x: torch.Tensor, dim: int):
    return (x - x.mean(dim=dim, keepdim=True)) / x.std(dim=dim, keepdim=True)


def cos_sin_features(time_steps: int):
    x = (2 * torch.pi / (time_steps - 1)) * torch.arange(0, time_steps)
    x = torch.stack((x.cos(), x.sin()))
    x = normalize(x, dim=-1)
    return x


def mackey_glass(time_steps: int, warmup: int = 500, reduce: int = 6, noise: float = 0.1,
                 beta: float = 0.2, gamma: float = 0.1, tau: int = 20, n: int = 10):
    x = torch.randn(tau + warmup + (time_steps * reduce)) * noise

    for i in range(tau, len(x) - 1):
        x[i + 1] += (1.0 - gamma) * x[i] + ((beta * x[i - tau]) / (1 + (x[i - tau] ** n)))

    x = x[tau + warmup:]
    x = x[torch.arange(0, len(x), reduce, dtype=torch.long)]
    x = x.unsqueeze(dim=0)
    x = normalize(x, dim=-1)
    return x


def get_causeme_data(experiment):
    """
    :return: Normalized Tensor of size (n_datasets, n_batches, n_nodes, T)
    """
    file_path = join_path(DATA_DIR, f"{experiment.split('.zip')[0]}.zip", make_dirs=False)
    with zipfile.ZipFile(file_path, "r") as f:
        # array with (n_exp, T, n_var)
        data = np.stack([np.loadtxt(f.open(name)) for name in sorted(f.namelist())])
    # (n_exp, T, n_var)
    data = torch.from_numpy(data).to(device=DEVICE)
    # standardize (n_exp, T, n_var)
    data = normalize(data, dim=1)
    # to float32 (n_exp, 1, n_var, T)
    return data.transpose(-1, -2).unsqueeze(dim=1).to(dtype=torch.float32)


"""
def get_dream_data(name, root='experiments/data/dream'):
    path = os.path.join(root, name if name.endswith('.zip') else name + '.zip')
    with zipfile.ZipFile(path, "r") as f:
        # array with (n_exp, T, n_var)
        data = pd.read_csv(f.open('data.tsv'), sep='\t').values[:, 1:]
        data = torch.from_numpy(data)
        # (T, n_exp, n_var)
        data = data.reshape((data.size()[0] // 21, 21, data.size()[1])).permute((1, 0, 2))
        # standardize
        data = (data - data.mean(dim=0)) / data.std(dim=0)
        # (n_exp, n_var, T)
        data = data.permute((1, 2, 0)).unsqueeze(dim=0)

        n_var = data.size()[-2]
        gt = []
        df = pd.read_csv(f.open('gt.txt'), sep='\t', names=['from', 'to', 'c'])
        for col in ['from', 'to']:
            df[col] = df[col].apply(lambda x: int(x[1:])-1)
        for _, (_from, to, _) in df[df['c'] != 0].iterrows():
            gt.append((_from, to))
    return data.to(device=DEVICE, dtype=torch.float32), sorted(gt, key=lambda x: x[0] * 100 + x[1])
"""


def toy_data_3_nodes(batch_size: int = 1, time_steps: int = 500, warmup: int = 200):
    """
    :return: Normalized Tensor of size (n_batches, 3_nodes, T), max_lags=1
    """
    assert batch_size > 0 and time_steps > 0 and warmup >= 0
    data = torch.randn((batch_size, 3, warmup + time_steps), dtype=torch.float64)
    for i in range(1, data.size()[-1]):
        data[:, 0, i] += torch.cos(data[:, 1, i - 1]) + torch.tanh(data[:, 2, i - 1])
        data[:, 1, i] += 0.35 * data[:, 1, i - 1] + data[:, 2, i - 1]
        data[:, 2, i] += torch.abs(0.5 * data[:, 0, i - 1]) + torch.sin(2 * data[:, 1, i - 1])
    data = data[..., warmup:]
    data = normalize(data, dim=-1)
    return data.to(device=DEVICE, dtype=torch.float32)


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
    return data.to(device=DEVICE, dtype=torch.float32)


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
    return data.to(device=DEVICE, dtype=torch.float32)


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
    return data.to(dtype=torch.float32), contributions.to(dtype=torch.float32)


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
    return data.to(device=DEVICE, dtype=torch.float32)


if __name__ == '__main__':
    ds = toy_data_coupled(time_steps=2000)
    ds = ds.cpu()
    _range = 80
    bs, n_var, T = ds.size()
    for v in range(n_var-1, -1, -1):
        plt.plot(ds[0, v, :_range], label=f"var {v}")
    plt.legend()
    plt.show()
    print()



