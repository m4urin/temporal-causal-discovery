import copy
import os

import numpy as np
from hyperopt import fmin, tpe, Trials, STATUS_OK

from typing import Any

from definitions import RESULTS_DIR
from src.data.temporal_causal_data import TemporalCausalData
from src.experiments.train_model import train_model
from src.models.config import ModelConfig, TrainConfig
from src.models.config.model_outputs import EvaluationResult
from src.utils.progress_3 import Nested_trange


def objective(params):
    """
    Objective function for hyperparameter tuning.

    Args:
        params (dict): A dictionary of hyperparameters.

    Returns:
        dict: A dictionary with the 'loss', 'status', and 'evaluation_result'.
    """
    evaluation_result = train_model(
        causal_data=params['causal_data'],
        model_config=ModelConfig(**params['model_space']),
        train_config=TrainConfig(**params['train_space']),
        progress_manager=params['progress_manager']
    )
    loss = -max(x['score'] for x in evaluation_result.train_result.aucroc_scores[-100:])  # maximize aucroc scores
    #loss = evaluation_result.train_result.auroc_scores.test_losses_true[-1]
    params['progress_manager'].update(loop_index=1)
    return {'loss': loss, 'status': STATUS_OK, 'evaluation_result': evaluation_result}


def merge_dicts(x, y):
    if y is None:
        return x

    # Create a deep copy of x to avoid modifying it directly
    result = copy.deepcopy(x)

    # Rest of the code remains the same
    for key, value in y.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


# model_type, max_evals, train_dataset, test_dataset, **overwrite
def run_hyperopt(max_evals: int,
                 causal_data: TemporalCausalData,
                 progress_manager: Nested_trange,
                 model_space: dict[str, Any],
                 train_space: dict[str, Any]) -> EvaluationResult:
    """
    Performs hyperparameter tuning on the given list of architectures using hyperopt.
    """
    save_dir = os.path.join(RESULTS_DIR, 'eval')
    os.makedirs(save_dir, exist_ok=True)

    space = {
        'causal_data': causal_data,
        'model_space': model_space,
        'train_space': train_space,
        'progress_manager': progress_manager
    }

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials, show_progressbar=False)

    best_trial = trials.trials[np.argmin([_trial['result']['loss'] for _trial in trials.trials])]
    evaluation_result: EvaluationResult = best_trial['result']['evaluation_result']

    return evaluation_result
