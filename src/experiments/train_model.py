import math

import torch
from torch.optim import AdamW
from tqdm import trange
from src.models.NAVAR import NAVAR
from src.models.TAMCaD import TAMCaD
from src.soft_roc_auc import roc_auc_score, soft_roc_auc_score
from src.utils import exponential_scheduler_with_warmup


def train_model(model: str, train_data: torch.Tensor, lr: float, epochs: int, weight_decay: float, test_size: float,
                true_causal_matrix: torch.Tensor = None, disable_tqdm: bool = False, lambda1: float = 0.2,
                beta: float = 0.2, start_coeff=-3, end_coeff=0, **model_params):
    """
    Trains a model.

    Args:
    - train_data: Input tensor of size (batch_size, n_variables, sequence_length).
    - lr: Learning rate.
    - epochs: Number of training epochs.
    - weight_decay: Weight decay for optimizer.
    - true_causal_matrix: Ground truth tensor of size (batch_size, n_variables, n_variables), default is None.
    - disable_tqdm: Boolean to control the display of progress bar.
    - lambda1: Hyperparameter for loss function.
    - kwargs: Additional arguments for NAVAR model.

    Returns:
    - A dictionary of training statistics.
    """

    if model == 'NAVAR':
        model = NAVAR(n_variables=train_data.size(1), **model_params)
    elif model == 'TAMCaD':
        model = TAMCaD(n_variables=train_data.size(1), **model_params)
    else:
        raise NotImplementedError('Not supported!')

    # Move model and data to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        train_data = train_data.cuda()
        if true_causal_matrix is not None:
            true_causal_matrix = true_causal_matrix.cuda()

    # Split data into training and testing sets
    x_test, y_test, gt_test, x_train, y_train, gt_train = split_data(train_data, true_causal_matrix, test_size)

    # Optimizers
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = exponential_scheduler_with_warmup(epochs, optimizer, start_factor=0.01, end_factor=0.05,
                                                  warmup_ratio=0.1, cooldown_ratio=0.2)

    stats = {
        'model': model,
        'params': {
            'model': model_params,
            'train': {'lr': lr, 'epochs': epochs, 'weight_decay': weight_decay, 'test_size': test_size}
        }
    }

    return execute_training(model, x_test, y_test, gt_test, x_train, y_train, gt_train, optimizer, scheduler,
                            epochs, stats, lambda1, beta, start_coeff, end_coeff, disable_tqdm)


def execute_training(model, x_test, y_test, gt_test, x_train, y_train, gt_train, optimizer, scheduler,
                     epochs, stats, lambda1, beta, start_coeff, end_coeff, disable_tqdm):
    stats.update({
        "train_phase": create_empty_stats_phase(),
        "test_phase": create_empty_stats_phase()
    })

    delta_coeff = end_coeff - start_coeff
    model.train()
    progressbar = trange(epochs, disable=disable_tqdm)

    for epoch in progressbar:
        coeff = 10 ** (start_coeff + delta_coeff * epoch / epochs)
        train_results = process_epoch(model, x_train, y_train, optimizer, scheduler, lambda1=lambda1, beta=beta, coeff=coeff)

        if epoch % 20 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                update_stats(model, stats, train_results, x_test, y_test, gt_test, gt_train,
                             coeff, lambda1, beta, progressbar, end=epoch == epochs - 1)

    return stats


def process_epoch(model, x, y, optimizer=None, scheduler=None, **kwargs):
    """Process one epoch for training/testing."""
    if optimizer:
        optimizer.zero_grad()

    output = model(x)
    loss = model.loss_function(y, **output, **kwargs)

    if optimizer:
        loss.backward()
        optimizer.step()
        scheduler.step()

    return loss.item(), output


def eval_epoch(model, model_output, gt=None):
    analysis = model.analysis(**model_output)
    temp_causal_matrix = analysis['temp_causal_matrix']
    auc = roc_auc_score(gt, temp_causal_matrix)[2].item() if gt is not None else None
    temp_conf_matrix, temp_ext_matrix, soft_auc = None, None, None
    if 'temp_confidence_matrix' in analysis:
        temp_conf_matrix = analysis['temp_confidence_matrix']
        soft_auc = soft_roc_auc_score(gt, temp_causal_matrix, temp_conf_matrix)[2].item() if gt is not None else None
    if 'temp_external_causal_matrix' in analysis:
        temp_ext_matrix = analysis['temp_external_causal_matrix']
    return temp_causal_matrix, temp_conf_matrix, temp_ext_matrix, auc, soft_auc


def split_data(data: torch.Tensor, true_causal_matrix: torch.Tensor = None, test_size: float = 0):
    """Splits data into training and testing sets."""
    # Check for valid test_size
    if not (0.0 <= test_size <= 1.0):
        raise ValueError("test_size should be between 0.0 and 1.0, inclusive.")

    # Function to split data or matrix
    def split_train_test(tensor_: torch.Tensor, s: int):
        return tensor_[..., :-s], tensor_[..., -s:]

    if test_size > 0.0:
        split_size = int(data.size(-1) * test_size)
        train_data, test_data = split_train_test(data, split_size)
        if true_causal_matrix is not None:
            train_causal, test_causal = split_train_test(true_causal_matrix, split_size)
            return test_data[..., :-1], test_data[..., 1:], test_causal[..., 1:], \
                train_data[..., :-1], train_data[..., 1:], train_causal[..., 1:]
        else:
            return test_data[..., :-1], test_data[..., 1:], None, \
                train_data[..., :-1], train_data[..., 1:], None

    if true_causal_matrix is not None:
        return None, None, None, data[..., :-1], data[..., 1:], true_causal_matrix[..., 1:]
    else:
        return None, None, None, data[..., :-1], data[..., 1:], None


def update_stats(model, stats, train_results, x_test, y_test, gt_test, gt_train,
                 coeff, lambda1, beta, progressbar, end=False):
    train_loss, train_output = train_results
    desc = f"train_loss={train_loss:.3f}"
    update_stats_phase(stats["train_phase"], model, train_output, gt_train, train_loss, end)

    if x_test is not None:
        test_loss, test_output = process_epoch(model, x_test, y_test, lambda1=lambda1, beta=beta, coeff=coeff)
        desc += f", test_loss={test_loss:.3f}"
        update_stats_phase(stats["test_phase"], model, test_output, gt_test, test_loss, end)

    progressbar.set_description(desc, refresh=False)


def update_stats_phase(phase_stats, model, model_output, gt, loss_value, end):
    phase_stats["loss"].append(loss_value)
    if gt is not None:
        causal_matrix, confidence_matrix, external_matrix, auc, soft_auc = eval_epoch(model, model_output, gt)
        phase_stats["auc"].append(auc)
        if confidence_matrix is not None:
            phase_stats["soft_auc"].append(soft_auc)
        update_best_score(phase_stats, causal_matrix, confidence_matrix, external_matrix, auc, soft_auc)
        update_end_score(phase_stats, causal_matrix, confidence_matrix, external_matrix, auc, soft_auc, end)
    elif end:
        causal_matrix, confidence_matrix, external_matrix, auc, soft_auc = eval_epoch(model, model_output)
        update_end_score(phase_stats, causal_matrix, confidence_matrix, external_matrix, auc, soft_auc, end)


def update_best_score(phase_stats, causal_matrix, confidence_matrix, external_matrix, auc, soft_auc):
    if confidence_matrix is not None:
        if soft_auc > phase_stats['best_score']:
            phase_stats['causal_matrix_best'] = causal_matrix
            phase_stats['conf_matrix_best'] = confidence_matrix
            phase_stats['best_score'] = soft_auc
            if external_matrix is not None:
                phase_stats['ext_matrix_best'] = external_matrix
    else:
        if auc > phase_stats['best_score']:
            phase_stats['causal_matrix_best'] = causal_matrix
            phase_stats['best_score'] = auc
            if external_matrix is not None:
                phase_stats['ext_matrix_best'] = external_matrix


def update_end_score(phase_stats, causal_matrix, confidence_matrix, external_matrix, auc, soft_auc, end):
    if end:
        phase_stats['causal_matrix_end'] = causal_matrix
        if confidence_matrix is not None:
            phase_stats['conf_matrix_end'] = confidence_matrix
        if auc:
            phase_stats['end_score'] = auc
        if soft_auc:
            phase_stats['end_score'] = soft_auc
        if external_matrix is not None:
            phase_stats['ext_matrix_best'] = external_matrix


def create_empty_stats_phase():
    return {
        "loss": [], "auc": [], "soft_auc": [],
        'best_score': -1, 'end_score': -1,
        'causal_matrix_best': None, 'causal_matrix_end': None,
        'conf_matrix_best': None, 'conf_matrix_end': None,
        'ext_matrix_best': None, 'ext_matrix_end': None,
    }
