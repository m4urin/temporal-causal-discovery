from time import time
import mlflow
import torch
from mlflow import MlflowClient
from torch.optim import AdamW
from tqdm import trange
from environment import GPULAB_JOB_ID
from src.training.experiment import get_all_metrics_params, save_artifacts
from src.utils import count_parameters, receptive_field, move_to


def train_model(
        experiment_name: str,
        model_type: type,
        dataset: dict,
        lr: float,
        epochs: int,
        weight_decay: float,
        test_size: float = 0.3,
        disable_tqdm: bool = False,
        experiment_run: str = None,
        **model_params):
    """
    Trains a PyTorch model and logs metrics to MLflow.

    Args:
        experiment_name (str): Name of the MLflow experiment.
        experiment_run (str): Name of the MLflow run.
        model_type (type): Type of the model to be trained.
        dataset (dict): Dictionary containing training and testing data.
        lr (float): Learning rate for optimizer.
        epochs (int): Number of training epochs.
        weight_decay (float): Weight decay for optimizer.
        test_size (float, optional): Proportion of data to use for testing. Defaults to 0.3.
        disable_tqdm (bool, optional): Disable tqdm progress bar. Defaults to False.
        **model_params: Additional model parameters.

    Returns:
        None
    """
    model = model_type(n_variables=dataset['data'].size(-2), **model_params)
    if torch.cuda.is_available():
        model = model.cuda()

    train_data, test_data, ground_truth = train_test_split(**dataset, test_size=test_size)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    experiment = MlflowClient().get_experiment_by_name(experiment_name)
    experiment_id = mlflow.create_experiment(experiment_name) if experiment is None else experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=experiment_run) as run:
        # train params

        result, time_per_epoch = execute_training(model, train_data, test_data, ground_truth, optimizer, epochs, disable_tqdm)

        log_params = {
            'model': str(model_type.__class__), **model_params,
            'lr': lr, 'epochs': epochs, 'weight_decay': weight_decay,
            'test_size': test_size, 'optimizer': 'AdamW',
            'n_params': count_parameters(model, trainable_only=True),
            'receptive_field': receptive_field(**model_params),
            'dataset_name': dataset['name'], 'training_time_per_epoch': time_per_epoch
        }
        mlflow.log_params(log_params)

        result.update({'model_params': log_params, 'train_data': train_data, 'test_data': test_data,
                       **get_all_metrics_params(run.info.run_id)})

        result = move_to(result, 'cpu')

        model = model.cpu()

        save_artifacts(experiment_name, experiment_run, model, result)

    mlflow.end_run()
    return result


def execute_training(model, train_data, test_data, ground_truth, optimizer, epochs, disable_tqdm):
    """
    Executes the training loop for the model.

    Args:
        model (torch.nn.Module): PyTorch model to be trained.
        train_data (dict): Training data dictionary.
        test_data (dict): Testing data dictionary.
        ground_truth (torch.Tensor): Ground truth data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int): Number of training epochs.
        disable_tqdm (bool): Disable tqdm progress bar.

    Returns:
        tuple: A tuple containing training artifacts and time per epoch.
    """
    model.train()
    disable_tqdm = disable_tqdm or GPULAB_JOB_ID is not None
    start_time = time()
    for epoch in (pbar := trange(epochs, disable=disable_tqdm)):
        if epoch == 2:
            start_time = time()
        optimizer.zero_grad()
        train_metrics, _ = model(**train_data, ground_truth=ground_truth)
        train_metrics['loss'].backward()
        optimizer.step()

        mlflow.log_metrics({f'train/{k}': v.item() for k, v in train_metrics.items()}, step=epoch)

        if epoch % 20 == 0 or epoch == epochs - 1:
            desc = ''
            if not disable_tqdm:
                desc += '[train] ' + ','.join([f'{k[:5]}={v.item():.2f}' for k, v in train_metrics.items()])

            if test_data is not None:
                model.eval()
                with torch.no_grad():
                    test_metrics, _ = model(**test_data, ground_truth=ground_truth)
                mlflow.log_metrics({f'test/{k}': v.item() for k, v in test_metrics.items()}, step=epoch)
                if not disable_tqdm:
                    desc += ' [test] ' + ','.join([f'{k[:5]}={v.item():.2f}' for k, v in test_metrics.items()])
                model.train()

            pbar.set_description(desc)

    time_per_epoch = (time() - start_time) / (epochs - 2)  # two epochs warmup

    model.eval()
    with torch.no_grad():
        _, train_artifacts = model(**train_data, ground_truth=ground_truth, create_artifacts=True)
        result = {'train_artifacts': train_artifacts}
        if test_data is not None:
            _, test_artifacts = model(**test_data, ground_truth=ground_truth, create_artifacts=True)
            result['test_artifacts'] = test_artifacts

    return result, time_per_epoch


def train_test_split(data: torch.Tensor, ground_truth: torch.Tensor = None,
                     data_noise_adjusted: torch.Tensor = None, test_size: float = 0, temporal_matrix=False, **kwargs):
    """
    Splits data into training and testing sets.

    Args:
        data (torch.Tensor): Input data to be split.
        ground_truth (torch.Tensor, optional): Ground truth data. Defaults to None.
        data_noise_adjusted (torch.Tensor, optional): Noise-adjusted data. Defaults to None.
        test_size (float, optional): Proportion of data to use for testing. Defaults to 0.

    Returns:
        tuple: A tuple containing training data, testing data, and ground truth data.
    """
    if not (0.0 <= test_size <= 1.0):
        raise ValueError("test_size should be between 0.0 and 1.0.")

    train, test = {'temporal_matrix': temporal_matrix}, {'temporal_matrix': temporal_matrix}

    n_train = int(data.size(-1) * (1.0 - test_size))
    n_test = data.size(-1) - n_train

    train['x'], test['x'] = data.split([n_train, n_test], dim=-1)

    if data_noise_adjusted is not None:
        train['x_noise_adjusted'], test['x_noise_adjusted'] = data_noise_adjusted \
            .split([n_train, n_test], dim=-1)

    # make contiguous for efficiency
    if torch.cuda.is_available():
        train = move_to(train, 'cuda')
        test = move_to(test, 'cuda')
        ground_truth = move_to(ground_truth, 'cuda')

    if test_size == 0.0:
        test = None

    return train, test, ground_truth
