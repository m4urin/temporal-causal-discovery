import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import AdamW

from src.utils.progress import iter_with_progress


def train_model(model, data, epochs, val_proportion, learning_rate, weight_decay, show_plot=False):

    # data is of size (batch_size, num_variables, sequence_length)
    batch_size, num_variables, sequence_length = data.size()

    n_test = int(val_proportion * sequence_length)
    n_train = sequence_length - n_test
    check_every = 30

    # to gpu if it is available
    if torch.cuda.is_available():
        model = model.cuda()
        data = data.cuda()

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_losses, val_losses = [], []

    x_train, y_train = data[..., :n_train - 1], data[..., 1:n_train]
    x_val, y_val = data[..., n_train:-1], data[..., n_train + 1:]

    train_data = data[..., :n_train], test_data = data[..., n_train:]

    model.train()
    for i in iter_with_progress(epochs):
        # predictions, contributions: (bs, num_nodes, time_steps), (bs, num_nodes, num_nodes, time_steps)
        result = model(train_data)
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
