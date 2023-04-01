import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import AdamW

from definitions import DEVICE
from src.models.navar import NAVAR
from src.utils import trange_mode


def train_model(model_constructor: type[NAVAR], data, epochs, val_proportion, learning_rate, lambda1,
                dropout, weight_decay, kernel_size, n_layers, hidden_dim,
                show_progress='tqdm', show_plot=False):
    assert show_progress in ['tqdm', 'console']
    # data: torch.Size([bs, num_nodes, T])
    bs, num_nodes, time_steps = data.size()
    n_test = int(val_proportion * time_steps)
    n_train = time_steps - n_test
    check_every = 50

    if hidden_dim > 8:
        raise Exception("hidden_dim should be <= 8, because hidden size is equal to 2^hidden_dim")

    model = model_constructor(num_nodes=num_nodes,
                              kernel_size=kernel_size,
                              n_layers=n_layers,
                              hidden_dim=2**hidden_dim,
                              lambda1=lambda1,
                              dropout=dropout).to(DEVICE)
    print('receptive field:', model.get_receptive_field())

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_losses, val_losses = [], []

    x_train, y_train = data[..., :n_train - 1], data[..., 1:n_train]
    x_val, y_val = data[..., n_train:-1], data[..., n_train + 1:]

    model.train()
    for i in trange_mode(epochs, show_progress):
        # predictions, contributions: (bs, num_nodes, time_steps), (bs, num_nodes, num_nodes, time_steps)
        loss = model.get_loss(x_train, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(loss.item())

        if n_test > 0 and (i % check_every == 0 or i == epochs - 1):
            model.eval()
            with torch.no_grad():
                val_losses.append(model.get_loss(x_val, y_val, sample=False).item())
            model.train()

    if show_plot:
        plt.clf()
        plt.plot(train_losses, label='train loss')
        plt.plot(np.arange(0, len(val_losses)) * check_every, val_losses, label='eval loss')
        plt.show()
        plt.clf()

    with torch.no_grad():
        result = model.evaluate(data, monte_carlo_dropout=20)

    if n_test > 0:
        # training on partial data
        skip_first = 1000 // check_every
        argmin = skip_first + np.argmin(val_losses[skip_first:])
        best_epochs = argmin * check_every
        return model, val_losses[argmin], best_epochs, result
    else:
        # training on complete data
        argmin = np.argmin(train_losses)
        best_epochs = argmin
        return model, train_losses[argmin], best_epochs, result
