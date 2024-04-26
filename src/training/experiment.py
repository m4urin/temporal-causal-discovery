import os
import shutil
from urllib.parse import urlparse

import torch
import mlflow
from mlflow import MlflowClient

from src.utils import OUTPUT_DIR


def save_artifacts(experiment_name, experiment_run, model, results):
    """
    Saves model and results artifacts to an MLflow experiment directory.

    Args:
        experiment_name (str): Name of the experiment.
        experiment_run (str): Name of the run.
        model (torch.nn.Module): PyTorch model to be saved.
        results (torch.Tensor): Results data to be saved.

    Returns:
        None
    """
    directory = os.path.join(OUTPUT_DIR, 'experiments', experiment_name)
    os.makedirs(directory, exist_ok=True)

    run_str = f'_{experiment_run}' if experiment_run is not None else ''
    results_path = os.path.join(directory, f'results{run_str}.pt')
    torch.save(results, results_path)
    #mlflow.log_artifact(results_path)

    model_path = os.path.join(directory, f'model{run_str}.pt')
    torch.save(model.cpu().state_dict(), model_path)
    #mlflow.log_artifact(model_path)


def get_all_metrics_params(run_id):
    """
    Retrieves all metrics and parameters associated with an MLflow run.

    Args:
        run_id: MLflow run object.

    Returns:
        dict: A dictionary containing train_metrics, test_metrics, and model_params.
    """
    client = MlflowClient()#.get_run(run.info.run_id)
    run = client.get_run(run_id)
    metrics_info = run.data.metrics.keys()
    all_metrics = {'train_metrics': {}, 'test_metrics': {}}
    for metric_name in metrics_info:
        m_terms = metric_name.split('/')
        phase = 'train_metrics' if m_terms[0] == 'train' else 'test_metrics'
        metrics_history = client.get_metric_history(run_id, metric_name)
        steps = torch.tensor([m.step for m in metrics_history])
        values = torch.tensor([m.value for m in metrics_history])
        all_metrics[phase][m_terms[1]] = (steps, values)
    return all_metrics


def clean_experiment(experiment_name):
    """
    Deletes a previous experiment with the same name if it exists and creates a new one.

    Args:
        experiment_name (str): Name of the experiment.

    Returns:
        str: Experiment ID of the new or existing experiment.
    """
    experiment = MlflowClient().get_experiment_by_name(experiment_name)
    if experiment is not None:
        experiment_id = experiment.experiment_id
        if experiment.lifecycle_stage != 'deleted':
            mlflow.delete_experiment(experiment_id)
        p = urlparse(mlflow.get_tracking_uri())
        shutil.rmtree(os.path.abspath(os.path.join(p.netloc, p.path, '.trash', experiment_id)))
    return mlflow.create_experiment(experiment_name)
