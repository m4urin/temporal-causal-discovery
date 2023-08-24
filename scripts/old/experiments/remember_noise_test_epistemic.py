import math

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from seaborn import heatmap
from tqdm import trange

from src.models.navar.navar_epistemic import NAVAR_TCN_E, loss_navar_e
from src.utils.pytorch import count_parameters


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def generate_random_data(batch_size=1, sequence_length=3000, scale1: str = 'reversed', scale2: str = 'scale_normal'):
    data = torch.randn(batch_size, 3, sequence_length + 1).clamp(min=-4.5, max=4.5)
    data[:, 0, np.random.choice(sequence_length+1, 4, replace=False)] = torch.tensor([-4.5, -4, 4, 4.5])

    true_function = torch.zeros_like(data)

    # var 1 noise:
    if scale1 == 'scale_normal':
        scale = 0.5 * torch.abs(data[:, 0, :-1])
    else:
        scale = (1 - torch.tanh(3 * data[:, 0, :-1]) ** 2) + 0.1
    data[:, 1, 1:] = scale * data[:, 1, 1:]

    # compute var 1
    for i in range(1, sequence_length + 1):
        true_function[:, 1, i] = np.sin(4 * data[:, 0, i - 1])
        data[:, 1, i] += true_function[:, 1, i]

    # var 2 noise:
    if scale2 == 'scale_normal':
        scale = 0.6 * torch.abs(data[0, 1, :-1])
    else:
        scale = (1 - torch.tanh(3 * data[0, 1, :-1]) ** 2) + 0.1
    data[0, 2, 1:] = scale * data[0, 2, 1:]

    # compute var 2
    for i in range(1, sequence_length + 1):
        true_function[:, 2, i] = 0.2 * data[:, 1, i - 1]
        data[:, 2, i] += true_function[:, 2, i]

    # normalize
    d_mean, d_std = data.mean(dim=-1, keepdim=True), data.std(dim=-1, keepdim=True)
    data = (data - d_mean) / d_std
    true_function = (true_function - d_mean) / d_std

    x = data[..., :-1]
    y = data[..., 1:]
    true_function = true_function[..., 1:]

    if torch.cuda.is_available():
        x, y, true_function = x.cuda(), y.cuda(), true_function.cuda()

    return x, y, true_function


def train_model(model, x, y, loss_function, epochs=3000, lr=1e-3, coeff_start=-5, coeff_end=0):
    coeff_delta = coeff_end - coeff_start
    optimizer = optim.AdamW(model.parameters(), lr=lr / 10, weight_decay=1e-6)

    losses = []

    pbar = trange(epochs)
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        result = model(x)
        loss = loss_function(y, *result, coeff=math.exp(coeff_start + (1 - epoch/epochs) * coeff_delta))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            pbar.set_description(f"Loss: {loss.item():.4f}")
        if epoch == int(0.1 * epochs):
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        if epoch == int(0.9 * epochs):
            optimizer = optim.AdamW(model.parameters(), lr=lr / 10)

    return model, losses


def post_training_analysis(model, x, y, true_function):
    with torch.no_grad():
        gamma, v, alpha, beta, _ = model(x)
        aleatoric = torch.sqrt(beta / v)
        epistemic = 1 / torch.sqrt(v)

    x = x[0, 0].cpu().numpy()
    indices = np.argsort(x)

    return \
        x[indices], \
        y[0, 1, indices].cpu().numpy(), \
        true_function[0, 1, indices].cpu().numpy(), \
        gamma[0, 1, indices].cpu().numpy(), \
        v[0, 1, indices].cpu().numpy(), \
        alpha[0, 1, indices].cpu().numpy(), \
        beta[0, 1, indices].cpu().numpy(), \
        aleatoric[0, 1, indices].cpu().numpy(), \
        epistemic[0, 1, indices].cpu().numpy()


def weighted_std(x, weights, dim):
    total_weight = weights.sum(dim=dim, keepdims=True)
    weighted_mean = (weights * x).sum(dim=dim, keepdims=True) / total_weight
    weighted_variance = (weights * (x - weighted_mean).pow(2)).sum(dim=dim) / total_weight.squeeze(dim)
    return torch.sqrt(weighted_variance)


def post_training_analysis_navar(model, x, y, true_function, from_var, to_var):
    with torch.no_grad():
        gamma, v, aleatoric_var, aleatoric_log_var, epistemic_var, gamma_prime, v_prime, aleatoric_var_prime, epistemic_var_prime = model(x)
        aleatoric = torch.sqrt(aleatoric_var)
        epistemic = torch.sqrt(epistemic_var)
        aleatoric_prime = torch.sqrt(aleatoric_var_prime)
        epistemic_prime = torch.sqrt(epistemic_var_prime)

        c1 = weighted_std(gamma + 2 * aleatoric_log_var, v, dim=-1)[0].t()
        c2 = weighted_std(gamma - 2 * aleatoric_log_var, v, dim=-1)[0].t()
        causal_matrix = (c1 + c2) / 2

        confidence_matrix = epistemic[0].mean(dim=-1).t()

    x = x[0, from_var].cpu().numpy()
    indices = torch.from_numpy(np.argsort(x)).long()

    return \
        x[indices], \
        y[0, to_var, indices].cpu().numpy(), \
        true_function[0, to_var, indices].cpu().numpy(), \
        gamma[0, from_var, to_var, indices].cpu().numpy(), \
        v[0, from_var, to_var, indices].cpu().numpy(), \
        aleatoric[0, from_var, to_var, indices].cpu().numpy(), \
        epistemic[0, from_var, to_var, indices].cpu().numpy(), \
        causal_matrix.cpu().numpy(), \
        confidence_matrix.cpu().numpy(), \
        gamma_prime[0, to_var, indices].cpu().numpy(), \
        aleatoric_prime[0, to_var, indices].cpu().numpy(), \
        epistemic_prime[0, to_var, indices].cpu().numpy()


def test():
    x_train, y_train, true_function_train = generate_random_data(sequence_length=2500)

    if torch.cuda.is_available():
        x_train, y_train, true_function_train = x_train.cuda(), y_train.cuda(), true_function_train.cuda()

    model = NAVAR_TCN_E(n_variables=3, hidden_dim=16, kernel_size=2, n_blocks=1, n_layers_per_block=2, dropout=0.0)
    print(model)

    if torch.cuda.is_available():
        model = model.cuda()

    print('n_parameters per var:', count_parameters(model) // 3)
    print('receptive field:', model.receptive_field)

    model = train_model(model, x_train, y_train, loss_navar_e, epochs=700, lr=1e-3, coeff_start=2e-2, coeff_end=1e-2)

    x, y, true_function, gamma, v, beta, aleatoric, epistemic, causal_matrix, confidence_matrix, gamma_prime, aleatoric_prime, epistemic_prime = post_training_analysis_navar(
        model, x_train, y_train, true_function_train, from_var=0, to_var=1)

if __name__ == '__main__':
    test()
