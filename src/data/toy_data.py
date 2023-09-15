import numpy as np
import torch


def f3(x0, x1, x2):
    return 0.8 * x0 + 0.8 * x1 - 0.8 * x2


def generate_random_data():
    batch_size, sequence_length = 1, 600
    data = (0.5 * torch.randn(batch_size, 4, sequence_length + 1))
    # data[:, 0, [100, 400, 200, 300]] = torch.tensor([-4.5, -4, 4, 4.5])
    data[:, -1] *= 0.0

    true_function = torch.zeros_like(data)

    # compute var 1
    for i in range(1, sequence_length + 1):
        true_function[:, 1, i] = np.sin(8 * data[:, 0, i - 1])
        data[:, 1, i] += true_function[:, 1, i]

    # compute var 2
    for i in range(1, sequence_length + 1):
        true_function[:, 2, i] = np.tanh(-4.0 * data[:, 1, i - 1])
        data[:, 2, i] += true_function[:, 2, i]

    # compute var 3
    for i in range(1, sequence_length + 1):
        true_function[:, 3, i] = np.sin(f3(*[data[:, g, i - 2] for g in range(3)]))
        data[:, 3, i] += true_function[:, 3, i]

    # normalize
    d_mean, d_std = data.mean(dim=-1, keepdim=True), data.std(dim=-1, keepdim=True)
    data = (data - d_mean) / d_std
    true_function = (true_function - d_mean) / d_std

    x = data[..., :-1]
    y = data[..., 1:]
    true_function = true_function[..., 1:]

    true_attn = torch.zeros(4, 4)
    true_attn[1, 0] = 1
    true_attn[2, 1] = 1
    true_attn[3, :3] = 1

    if torch.cuda.is_available():
        x, y, true_function = x.cuda(), y.cuda(), true_function.cuda()

    return x, y, true_function, true_attn

