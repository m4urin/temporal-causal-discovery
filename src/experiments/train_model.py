from time import time

import torch
from sklearn import metrics
from torch.optim import AdamW
from tqdm import trange

from src.models.NAVAR import NAVAR
from src.models.TAMCaD import TAMCaD

from src.utils.pytorch import exponential_scheduler_with_warmup


def train_NAVAR(train_data: torch.Tensor, lr, epochs, weight_decay, test_size,
                true_causal_matrix=None, disable_tqdm=False, lambda1=0.2, **kwargs):
    """
    data: tensor, is of size (batch_size, n_variables, sequence_length)
    lr: float, learning rate
    epochs: int
    weight_decay: float
    true_causal_matrix: tensor, is of size (batch_size, n_variables, n_variables)
    disable_tqdm: bool, turn off while doing hyperopt search
    """

    if true_causal_matrix is not None:
        true_causal_matrix = true_causal_matrix.flatten()

    model = NAVAR(n_variables=train_data.size(1), **kwargs)

    # Move the model and data to the GPU if it's available
    if torch.cuda.is_available():
        model = model.cuda()
        train_data = train_data.cuda()

    x_test, y_test = None, None
    if test_size > 0.0:
        n = int(train_data.size(-1) * test_size)
        test_data = train_data[..., -n:]
        train_data = train_data[..., :-n]
        x_test, y_test = test_data[..., :-1], test_data[..., 1:]
    x_train, y_train = train_data[..., :-1], train_data[..., 1:]

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = exponential_scheduler_with_warmup(optimizer,
                                                  start_factor=0.05, end_factor=0.05,
                                                  warmup_iters=int(epochs * 0.05),
                                                  cooldown_iters=int(epochs * 0.2),
                                                  total_iters=epochs)

    train_losses, test_losses, train_auc, test_auc, train_cm, test_cm = [], [], [], [], [], []

    start_time = time()

    model.train()

    progressbar = trange(epochs, disable=disable_tqdm)
    for _ in progressbar:
        loss, auc, cm = navar_epoch(model, x_train, y_train, lambda1, true_causal_matrix)

        train_losses.append(loss.item())
        train_cm.append(cm)
        if true_causal_matrix is not None:
            train_auc.append(auc)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        desc = "train_loss={:.3f}, train_auc={:.2f}".format(train_losses[-1], train_auc[-1])

        if test_size > 0.0:
            with torch.no_grad():
                loss, auc, cm = navar_epoch(model, x_test, y_test, lambda1, true_causal_matrix)

                test_losses.append(loss.item())
                test_cm.append(cm)
                if true_causal_matrix is not None:
                    test_auc.append(auc)
                desc += "test_loss={:.3f}, test_auc={:.2f}".format(test_losses[-1], test_auc[-1])

        progressbar.set_description(desc, refresh=False)

    return {
        ...
    }


def navar_epoch(model, x, y, lambda1, true_causal_matrix_flat=None):
    output = model(x)
    loss = model.loss_function(y, *output, lambda1=lambda1)
    contributions = output[1]
    causal_matrix = contributions.detach().std(dim=-1).cpu().numpy()  # (batch_size, n_variables, n_variables)
    if true_causal_matrix_flat is not None:
        fpr, tpr, _ = metrics.roc_curve(true_causal_matrix_flat, causal_matrix.flatten())
        auc = metrics.auc(fpr, tpr)
    else:
        auc = None
    return loss, auc, causal_matrix
