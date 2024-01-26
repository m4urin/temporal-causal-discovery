import numpy as np
import torch
from hyperopt import fmin, tpe, Trials, STATUS_OK

from src.training.train_model import train_model
from src.utils import ConsoleProgressBar


global_best_loss = float('inf')


def run_hyperopt(dataset: dict, max_evals: int, subset_evals: int, **search_space) -> dict:
    """
    Executes the hyperparameter optimization process.

    :param max_evals: Maximum evaluations for optimization.
    :param subset_evals: Number of subset evaluations.
    :param search_space: Search space for optimization.
    :param dataset: Dataset containing: name, data, data_noise_adjusted(optional), ground_truth(optional)

    :return: Best training parameters.
    """
    global global_best_loss
    global_best_loss = float('inf')

    num_datasets = dataset['data'].size(0)
    subset_evals = min(num_datasets, subset_evals)

    hp_space = {
        'dataset': dataset,
        'subset_evals': subset_evals,
        'pbar': ConsoleProgressBar(total=max_evals * subset_evals, title='Hyperopt'),
        **search_space
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


def objective(dataset: dict, subset_evals: int, pbar: ConsoleProgressBar, **train_params) -> dict:
    """
    Objective function for hyperopt optimization.

    :param dataset: The dataset for optimization.
    :param pbar: Progress bar for tracking.
    :param subset_evals: Number of subset evaluations.
    :param train_params: Training parameters for the model.

    :return: Dictionary containing optimization results.
    """
    global global_best_loss
    total_loss = 0
    metric = None

    for i in range(subset_evals):
        subset_data = {k: v[i] if isinstance(v, torch.Tensor) else v for k, v in dataset.items()}

        result = train_model(dataset=subset_data, **train_params)
        metric, score = get_score(result)
        total_loss += score

        if i < subset_evals - 1:
            pbar.update(desc=f"Subset {i+1}/{subset_evals}")

    total_loss /= subset_evals

    if total_loss < global_best_loss:
        global_best_loss = total_loss

    desc = f"Subset {subset_evals}/{subset_evals}, " if subset_evals > 1 else ""
    pbar.update(desc=f"{desc}loss: {total_loss:.3f}, best_loss: {global_best_loss:.3f} ({metric})")

    return {
        'loss': total_loss,
        'status': STATUS_OK,
        'train_params': train_params
    }


def get_score(result):
    phase = 'test_metrics' if len(result['test_metrics']) > 0 else 'train_metrics'
    # metrics ordered by preference
    for k, metric_name in [(-1, 'soft_AUROC'), (-1, 'AUROC'), (1, 'noise_adjusted_regression_loss'), (1, 'loss')]:
        if metric_name in result[phase]:
            _, scores = result[phase][metric_name]
            return metric_name, torch.min(k * scores[-5:]).item()
