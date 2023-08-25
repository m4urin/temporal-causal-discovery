import numpy as np
import torch
from hyperopt import fmin, tpe, Trials, STATUS_OK
from src.training.train_model import train_model
from src.utils import ConsoleProgressBar


def compute_loss_for_dataset(dataset: dict, train_params: dict) -> float:
    """
    Computes the loss for a given dataset and training parameters.

    :param dataset: The dataset to compute the loss for.
    :param train_params: Training parameters for the model.

    :return: Computed loss for the given dataset.
    """
    _, train_stats = train_model(dataset=dataset, **train_params)
    auc_test = train_stats["test_phase"]["auc"]
    loss_test = train_stats["test_phase"]["loss"]
    auc_train = train_stats["train_phase"]["auc"]
    loss_train = train_stats["train_phase"]["loss"]

    # Use AUC if available, otherwise use loss
    if len(auc_test) > 0:
        return -max(auc_test[-5:])  # max AUC
    if len(loss_test) > 0:
        return min(loss_test[-5:])  # min loss
    if len(auc_train) > 0:
        return -max(auc_train[-5:])  # max AUC
    return min(loss_train[-5:])  # min loss


global_best_loss = float('inf')


def objective(dataset: dict, pbar: ConsoleProgressBar, subset_evals: int, **train_params) -> dict:
    """
    Objective function for hyperopt optimization.

    :param dataset: The dataset for optimization.
    :param pbar: Progress bar for tracking.
    :param subset_evals: Number of subset evaluations.
    :param train_params: Training parameters for the model.

    :return: Dictionary containing optimization results.
    """
    global global_best_loss
    num_datasets = dataset['data'].size(0)
    total_loss = 0
    n_evals = min(num_datasets, subset_evals)

    for i in range(n_evals):
        total_loss += compute_loss_for_dataset(dataset_subset(dataset, i), train_params)
        if i < n_evals - 1:
            pbar.update(desc=f"Subset {i+1}/{n_evals}")

    if total_loss < global_best_loss:
        global_best_loss = total_loss

    pbar.update(desc=f"Subset {n_evals}/{n_evals}, loss: {total_loss:.3f}, best_loss: {global_best_loss:.3f}")

    return {
        'loss': total_loss,
        'status': STATUS_OK,
        'train_params': train_params
    }


def run_hyperopt(max_evals: int, subset_evals: int, **space) -> dict:
    """
    Executes the hyperparameter optimization process.

    :param max_evals: Maximum evaluations for optimization.
    :param space: Search space for optimization.

    :return: Best training parameters.
    """
    hp_space = {
        'subset_evals': subset_evals,
        'pbar': ConsoleProgressBar(total=max_evals * subset_evals, title='Hyperopt'),
        **space
    }
    trials = Trials()

    fmin(fn=lambda params: objective(**params),
         space=hp_space,
         algo=tpe.suggest,
         max_evals=max_evals,
         trials=trials,
         show_progressbar=False)

    idx_best_score = np.argmin([trial['result']['loss'] for trial in trials.trials])
    best_train_params = trials.trials[idx_best_score]['result']['train_params']

    return best_train_params


def dataset_subset(dataset: dict, i: int):
    return {k: v[i] if isinstance(v, torch.Tensor) else v for k, v in dataset.items()}
