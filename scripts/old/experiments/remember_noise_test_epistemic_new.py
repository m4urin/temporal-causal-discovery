import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
from scripts.old.experiments.EDL_losses.losses import EvidentialRegression
from src.models.NAVAR.navar_epistemic import NAVAR_TCN_EPISTEMIC
from src.utils.pytorch import count_parameters


def initialize_model():
    model = NAVAR_TCN_EPISTEMIC(n_variables=3, hidden_dim=12, kernel_size=2, n_blocks=1, n_layers_per_block=2)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def generate_random_data(scale1: str = 'reversed', scale2: str = 'scale_normal'):
    batch_size, sequence_length = 1, 3000
    data = torch.randn(batch_size, 3, sequence_length + 1).clamp(min=-4.5, max=4.5)
    data[:, 0, [500, 1500, 1000, 2000]] = torch.tensor([-4.5, -4, 4, 4.5])

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
        scale = 0.5 * torch.abs(data[0, 1, :-1])
    else:
        scale = (1 - torch.tanh(3 * data[0, 1, :-1]) ** 2) + 0.1
    data[0, 2, 1:] = scale * data[0, 2, 1:]

    # compute var 2
    for i in range(1, sequence_length + 1):
        true_function[:, 2, i] = np.tanh(-0.7 * data[:, 1, i - 1])
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


def train_model(model, x, y, epochs=3000, lr=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr / 10, weight_decay=1e-4)

    pbar = trange(epochs)
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        gamma, v, alpha, beta, gamma_contributions = model(x)

        evident_loss = EvidentialRegression(y, gamma, v, alpha, beta, coeff=1e-2)
        regularization_loss = gamma_contributions.abs().mean()
        loss = evident_loss + 0.2 * regularization_loss

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            pbar.set_description(f"Loss: {loss.item():.4f}")
        if epoch == int(0.2 * epochs):
            optimizer = optim.Adam(model.parameters(), lr=lr)
        if epoch == int(0.8 * epochs):
            optimizer = optim.Adam(model.parameters(), lr=lr / 10)

    return model


def convert_data(x, y, true_function, gamma, alpha, v, beta, aleatoric, epistemic):
    x = x[0, 0].cpu().numpy()
    indices = np.argsort(x)

    return \
        x[indices], \
        y[0, 1, indices].cpu().numpy(), \
        true_function[0, 1, indices].cpu().numpy(), \
        gamma[0, 1, indices].cpu().numpy(), \
        alpha[0, 1, indices].cpu().numpy(), \
        v[0, 1, indices].cpu().numpy(), \
        beta[0, 1, indices].cpu().numpy(), \
        aleatoric[0, 1, indices].cpu().numpy(), \
        epistemic[0, 1, indices].cpu().numpy()


def main():
    model = initialize_model()
    x, y, true_function = generate_random_data()

    print('n_parameters per var:', count_parameters(model) // 3)
    print('receptive field:', model.receptive_field)

    model = train_model(model, x, y)
    model.eval()

    with torch.no_grad():
        gamma, v, alpha, beta, _ = model(x)
        aleatoric = np.sqrt(beta * (1 + v) / (alpha * v))
        epistemic = 1 / np.sqrt(v)

    results = convert_data(x, y, true_function, gamma, alpha, v, beta, aleatoric, epistemic)
    plot_results(*results)


if __name__ == '__main__':
    main()
