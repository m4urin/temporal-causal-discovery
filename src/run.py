import os
from pprint import pprint

import torch
from hyperopt.pyll.stochastic import sample
from src.training.hyper_optimization import run_hyperopt
from src.training.train_model import train_model
from src.utils import load_data, read_json, write_json, ConsoleProgressBar, get_method_simple_description


def get_train_params(dataset: dict, output_dir: str, max_evals: int, space: dict) -> dict:
    """Determine training parameters either from hyperopt, prior results or sample space."""
    params_path = os.path.join(output_dir, 'best_train_params.json')

    if os.path.exists(params_path):
        print(f'Best parameters already present.. Using {params_path}:')
        train_params = read_json(params_path)
        pprint(train_params)
        return train_params
    elif max_evals > 0:
        print(f'Run hyperopt to find best parameters..')
        train_params = run_hyperopt(max_evals, dataset=dataset, **space)
        print("Best parameters found:")
        pprint(train_params)
        print(f'Write best parameters to {params_path}.')
        write_json(params_path, train_params)
        return train_params
    else:
        print(f'max_evals is smaller than 1. Use the provided space and sample if necessary.')
        train_params = sample(space)
        pprint(train_params)
        return sample(space)


def evaluate_using_params(dataset: dict, output_dir: str, train_params: dict) -> list:
    """Evaluate the model using the provided dataset and training parameters."""
    results_path = os.path.join(output_dir, 'results.pt')

    if os.path.exists(results_path):
        return torch.load(results_path)

    results = []
    train_params['test_size'] = 0.0  # Set to 0% test_size for evaluation

    n_datasets = dataset['data'].size(0)
    pbar = ConsoleProgressBar(total=n_datasets, title='Run complete with best parameters')
    for i in range(n_datasets):
        _, train_stats = train_model(
            dataset=dataset_subset(dataset, i),
            **train_params)

        results.append(train_stats)
        pbar.update()

    torch.save(results, results_path)

    return results


def get_output_dir(dataset_name: str, **model_params) -> str:
    """Generate output directory based on current date, time, dataset name, and GPULAB job ID."""
    dir_name = f"{get_method_simple_description(**model_params)}_{dataset_name}"
    full_path = os.path.join(OUTPUT_DIR, dir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def run(data_dir: str, dataset_name: str, max_evals: int, **space) -> list:
    """
    Performs hyperparameter tuning on the given list of architectures using hyperopt.
    """
    output_dir = get_output_dir(dataset_name, **space)

    dataset = load_data(data_dir, dataset_name)
    train_params = get_train_params(dataset, output_dir, max_evals, space)
    results = evaluate_using_params(dataset, output_dir, train_params)

    return results
