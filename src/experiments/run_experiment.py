import gzip
import os
import pickle
from typing import Type, List

from definitions import RESULTS_DIR
from src.data.dataset import Dataset
from src.experiments.hyper_optimization import run_hyperopt
from src.models.temporal_causal_model import TemporalCausalModel


def run_experiment(name: str, dataset: Dataset, architectures: List[Type[TemporalCausalModel]], max_evals=100):
    for model_type in architectures:
        eval_result = run_hyperopt(dataset, model_type, max_evals=max_evals)
        with gzip.open(os.path.join(RESULTS_DIR, f"experiments/{name}/{architectures}.pkl.gz"), "wb") as f:
            pickle.dump(eval_result, f)
