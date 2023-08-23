import gzip
import os
import pickle
from typing import Any

from definitions import RESULTS_DIR
from src.data.temporal_causal_data import TemporalCausalData
from src.experiments.hyper_optimization import run_hyperopt


def run_experiment(name: str,
                   max_evals: int,
                   causal_data: TemporalCausalData,
                   model_spaces: list[dict[str, Any]],
                   train_space: dict[str, Any]):
    dir_path = os.path.join(RESULTS_DIR, "experiments", name)

    all_results = []
    for model_space in model_spaces:
        file_path = os.path.join(dir_path, f"{model_space['name']}.pkl.gz")

        if os.path.exists(file_path):
            with gzip.open(file_path, "rb") as f:
                eval_result = pickle.load(f)
        else:
            eval_result = run_hyperopt(max_evals, causal_data, model_space, train_space)
            os.makedirs(dir_path, exist_ok=True)
            with gzip.open(file_path, "wb") as f:
                pickle.dump(eval_result, f)
        all_results.append(eval_result)

    return all_results
