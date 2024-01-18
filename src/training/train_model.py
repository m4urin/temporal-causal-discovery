import os.path

import mlflow
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import AdamW
from tqdm import trange

from environment import GPULAB_JOB_ID, OUTPUT_DIR
from src.data.visualisations import plot_multi_roc_curve, plot_heatmaps, plot_contemporaneous_relationships
from src.utils import count_parameters, receptive_field

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(
        model_type: type,
        dataset: dict,
        lr: float,
        epochs: int,
        weight_decay: float,
        test_size: float = 0.3,
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
    - kwargs: Additional arguments for NAVAR model.

    Returns:
    - A dictionary of training statistics.
    """
    model = model_type(**model_params).to(DEVICE)
    train_data, test_data, ground_truth = train_test_split(**dataset, test_size=test_size)

    # Optimizers
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    with mlflow.start_run():
        # train params
        mlflow.log_params({
            'model': str(model_type), **model_params,
            'lr': lr, 'epochs': epochs, 'weight_decay': weight_decay,
            'test_size': test_size, 'optimizer': 'AdamW',
            'n_params': count_parameters(model, trainable_only=True),
            'receptive_field': receptive_field(**model_params)
        })

        execute_training(model, train_data, test_data, ground_truth, optimizer, epochs, disable_tqdm)

        with torch.no_grad():
            model.eval()
            create_artifacts(model, test_data)


def execute_training(model, train_data, test_data, ground_truth, optimizer, epochs, disable_tqdm):
    for epoch in (pbar := trange(epochs, disable=disable_tqdm or GPULAB_JOB_ID is not None)):
        model.train()
        optimizer.zero_grad()
        output = model(**train_data)
        output['loss'].backward()
        optimizer.step()
        mlflow.log_metrics(output)

        if epoch >= 20 and (epoch % 20 == 0 or epoch == epochs - 1):
            with torch.no_grad():
                model.eval()
                desc = eval_step(epoch, model, **train_data, lambda1=lambda1)
                if test_data is not None:
                    desc += ', ' + eval_step(epoch, model, **test_data, lambda1=lambda1)
            pbar.set_description(desc)



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

        auroc = auroc(gt, analysis['temp_causal_matrix'])[2].item()
        mlflow.log_metric(f'AUROC/{phase}', auroc, step=epoch)
        info[f'{phase}_auroc'] = auroc

        if 'temp_confidence_matrix' in analysis:
            soft_auroc = soft_AUROC(gt, analysis['temp_causal_matrix'],
                                    analysis['temp_confidence_matrix'])[2].item()
            mlflow.log_metric(f'Soft-AUROC/{phase}', soft_auroc, step=epoch)
            info[f'{phase}_softauroc'] = soft_auroc

    description = ', '.join([f"{k}={v:.2f}" for k, v in info.items()])

    return description


def create_artifacts(model, test_data):
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



def train_test_split(data: torch.Tensor, ground_truth: torch.Tensor = None,
                     data_noise_adjusted: torch.Tensor = None, test_size: float = 0):
    """Splits data into training and testing sets."""
    # Check for valid test_size
    if not (0.0 <= test_size <= 1.0):
        raise ValueError("test_size should be between 0.0 and 1.0.")

    train, test = {'phase': 'train'}, {'phase': 'test'}

    n_train = int(data.size(-1) * (1.0 - test_size))
    n_test = data.size(-1) - n_train

    train['x'], test['x'] = data.split([n_train, n_test], dim=-1)

    if data_noise_adjusted:
        train['x_noise_adjusted'], test['x_noise_adjusted'] = data_noise_adjusted \
            .split([n_train, n_test], dim=-1)

    # make contiguous for efficiency
    if torch.cuda.is_available():
        train = dict_to_cuda(train)
        test = dict_to_cuda(test)
        ground_truth = ground_truth.cuda()

    if test_size == 0.0:
        test = None

    return train, test, ground_truth


def dict_to_cuda(a_dict):
    return {k: v.contiguous().cuda() if isinstance(v, torch.Tensor) else v for k, v in a_dict.items()}
