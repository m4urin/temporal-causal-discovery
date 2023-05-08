import os
import pickle
import gzip

import numpy as np
import torch
from hyperopt import fmin, tpe, Trials, STATUS_OK

from typing import List, Type

from definitions import RESULTS_DIR
from src.data.dataset import Dataset
from src.models.temporal_causal_model import TemporalCausalModel
from src.training.train_model import train_model
from src.utils.config import ModelConfig, TrainConfig
from src.utils.model_outputs import EvaluationResult


def objective(params):
    """
    Objective function for hyperparameter tuning.

    Args:
        params (dict): A dictionary of hyperparameters.

    Returns:
        dict: A dictionary with the 'loss', 'status', and 'evaluation_result'.
    """
    model_config = ModelConfig(**params)
    train_config = TrainConfig(**params)

    try:
        evaluation_result = train_model(params['dataset'], model_config, train_config, show_progress=True)
        loss = evaluation_result.train_result.test_losses[-1]
    except Exception as e:
        print(f"Exception while training model: {e}")
        evaluation_result = None
        loss = np.inf
        raise e

    return {'loss': loss, 'status': STATUS_OK, 'evaluation_result': evaluation_result}


def run_hyperopt(dataset: Dataset, architectures: List[Type[TemporalCausalModel]],
                 max_evals: int = 100) -> List[EvaluationResult]:
    """
    Performs hyperparameter tuning on the given list of architectures using hyperopt.

    Args:
        dataset (torch.Tensor): The input data.
        architectures (List[Type[TemporalCausalModel]]): A list of PyTorch model class types.
        max_evals (int): The maximum number of evaluations.

    Returns:
        List[EvaluationResult]: A list of EvaluationResult objects, one for each architecture.
    """
    save_dir = os.path.join(RESULTS_DIR, 'eval')
    os.makedirs(save_dir, exist_ok=True)

    results = []

    for model_type in architectures:
        space = {
            "dataset": dataset,
            "num_variables": dataset.num_variables,
            **model_type.get_hp_space(),
            **TrainConfig.get_hp_space(val_proportion=0.3)
        }

        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials, show_progressbar=False)

        best_trial = sorted(trials.trials, key=lambda x: x['result']['loss'])[0]
        evaluation_result: EvaluationResult = best_trial['result']['evaluation_result']

        # Save the evaluation result to a file
        with gzip.open(os.path.join(save_dir, f'{evaluation_result.model_config.url}.pkl.gz'), "wb") as f:
            pickle.dump(evaluation_result, f)

        # Add the evaluation result to the list
        results.append(evaluation_result)

    return results
