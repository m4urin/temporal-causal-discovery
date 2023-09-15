import os.path

import mlflow
import torch
from matplotlib import pyplot as plt
from torch.optim import AdamW
from tqdm import trange

from config import GPULAB_JOB_ID, OUTPUT_DIR
from src.data.visualisations import plot_multi_roc_curve, plot_heatmaps, plot_contemporaneous_relationships
from src.models.navar import NAVAR
from src.models.TAMCaD import TAMCaD
from src.eval.soft_roc_auc import roc_auc_score, soft_roc_auc_score
from src.utils import exponential_scheduler_with_warmup, augment_with_sine, count_parameters


def train_model(
        dataset: dict,
        model: str,
        lr: float,
        epochs: int,
        weight_decay: float,
        test_size: float,
        lambda1: float,
        disable_tqdm: bool = False,
        **model_params):
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
    model = instantiate_model(model, **model_params, num_variables=dataset['data'].size(-2))
    all_data, train_data, test_data = prepare_data(**dataset, test_size=test_size)

    # Optimizers
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = exponential_scheduler_with_warmup(epochs, optimizer, start_factor=0.01, end_factor=0.05,
                                                  warmup_ratio=0.1, cooldown_ratio=0.2)
    with mlflow.start_run():
        # train params
        mlflow.log_params({'model': model, **model_params, 'lr': lr, 'epochs': epochs, 'weight_decay': weight_decay,
                           'test_size': test_size, 'lambda1': lambda1, 'optimizer': 'AdamW'})
        mlflow.log_metric('n_params', count_parameters(model, trainable_only=True))

        execute_training(model, train_data, test_data, optimizer, scheduler, epochs, lambda1, disable_tqdm)

        with torch.no_grad():
            model.eval()
            create_artifacts(model, **all_data, lambda1=lambda1)


def execute_training(model, train_data, test_data, optimizer, scheduler, epochs, lambda1, disable_tqdm):
    for epoch in (pbar := trange(epochs, disable=disable_tqdm or GPULAB_JOB_ID is not None)):
        model.train()
        epoch_step(model, **train_data, optimizer=optimizer, scheduler=scheduler, lambda1=lambda1)

        if epoch >= 20 and (epoch % 20 == 0 or epoch == epochs - 1):
            with torch.no_grad():
                model.eval()
                desc = eval_step(epoch, model, **train_data, lambda1=lambda1)
                if test_data is not None:
                    desc += ', ' + eval_step(epoch, model, **test_data, lambda1=lambda1)
            pbar.set_description(desc)


def epoch_step(model, x, y, lambda1, optimizer=None, scheduler=None, **kwargs):
    if optimizer:
        optimizer.zero_grad()

    output = model(x)
    loss = model.loss_function(y, **output, lambda1=lambda1)

    if optimizer:
        loss.backward()
        optimizer.step()
        scheduler.step()

    return output, loss.item()


def eval_step(epoch, model, x, y, gt, y_mean, phase, lambda1):
    info = {}

    output = model(x)
    loss = model.loss_function(y, **output, lambda1=lambda1)
    loss_mean = model.loss_function(y_mean, **output, lambda1=lambda1)

    mlflow.log_metric(f'loss/{phase}', loss, step=epoch)
    mlflow.log_metric(f'loss_mean/{phase}', loss_mean, step=epoch)
    info[f'{phase}_loss'] = loss

    if gt is not None:
        analysis = model.analysis(**output)

        auroc = roc_auc_score(gt, analysis['temp_causal_matrix'])[2].item()
        mlflow.log_metric(f'AUROC/{phase}', auroc, step=epoch)
        info[f'{phase}_auroc'] = auroc

        if 'temp_confidence_matrix' in analysis:
            soft_auroc = soft_roc_auc_score(gt, analysis['temp_causal_matrix'],
                                            analysis['temp_confidence_matrix'])[2].item()
            mlflow.log_metric(f'Soft-AUROC/{phase}', soft_auroc, step=epoch)
            info[f'{phase}_softauroc'] = soft_auroc

    description = ', '.join([f"{k}={v:.2f}" for k, v in info.items()])

    return description


def create_artifacts(model, x, y, gt, lambda1):
    artifact_path = os.path.join(OUTPUT_DIR, '.artifacts')
    os.makedirs(artifact_path, exist_ok=True)

    def log_pt(data, filename):
        file_path = os.path.join(artifact_path, filename)
        torch.save(data, file_path)
        mlflow.log_artifact(file_path)

    def log_fig(filename):
        mlflow.log_figure(plt.gcf(), filename)
        plt.clf()

    output, _ = epoch_step(model, x, y, lambda1)
    analysis = model.analysis(**output)
    log_pt(analysis, 'analysis.pt')

    if gt is not None:
        plot_multi_roc_curve(gt, [analysis['temporal_causal_matrix']], names=[model.__class__.__name__])
        log_fig('ROC.png')

    matrices = [analysis['temporal_causal_matrix']]
    names = ['Prediction']
    if gt is not None:
        matrices.insert(0, gt)
        names.insert(0, 'Ground truth')

    plot_contemporaneous_relationships(*matrices, causal_links=[(0, 1), (1, 2), (2, 0)], names=names, smooth=0.5)
    log_fig('contemporaneous.png')

    if 'temp_confidence_matrix' in analysis and analysis['temp_confidence_matrix'] is not None:
        matrices += [analysis['temp_confidence_matrix']]
        names += ['Confidence']

    matrices = [m.mean(dim=-1) for m in matrices]
    plot_heatmaps(*matrices, names=names)
    log_fig('matrix.png')

    os.remove(artifact_path)


def instantiate_model(model_name: str, **model_params):
    if model_name == 'NAVAR':
        model = NAVAR(**model_params)
    elif model_name == 'TAMCaD':
        model = TAMCaD(**model_params)
    else:
        raise NotImplementedError('Not supported!')

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def prepare_data(data: torch.Tensor, ground_truth: torch.Tensor = None,
                 data_mean: torch.Tensor = None, test_size: float = 0):
    """Splits data into training and testing sets."""
    # Check for valid test_size
    if not (0.0 <= test_size <= 1.0):
        raise ValueError("test_size should be between 0.0 and 1.0, inclusive.")

    all_data = {
        'x': data[..., :-1],
        'y': data[..., 1:],
        'gt': ground_truth[..., 1:] if ground_truth is not None else None,
        'y_mean': data_mean[..., 1:] if data_mean is not None else None
    }

    train, test = {'phase': 'train'}, {'phase': 'test'}

    if test_size > 0.0:
        split_size = max(1, int(data.size(-1) * test_size)) + 1
        train['x'] = data[..., :-split_size - 1]
        train['y'] = data[..., 1:-split_size]
        test['x'] = data[..., -split_size:-1]
        test['y'] = data[..., 1 - split_size:]
        train['gt'] = ground_truth[..., 1:-split_size] if ground_truth is not None else None
        test['gt'] = ground_truth[..., 1 - split_size:] if ground_truth is not None else None
        train['y_mean'] = data_mean[..., 1:-split_size] if data_mean is not None else None
        test['y_mean'] = data_mean[..., 1 - split_size:] if data_mean is not None else None

    # make contiguous for efficiency
    if torch.cuda.is_available():
        all_data = dict_to_cuda(all_data)
        train = dict_to_cuda(train)
        test = dict_to_cuda(test)

    if test_size <= 0.0:
        train.update(all_data)
        test = None

    return all_data, train, test


def dict_to_cuda(a_dict):
    return {k: v.contiguous().cuda() if isinstance(v, torch.Tensor) else v for k, v in a_dict.items()}
