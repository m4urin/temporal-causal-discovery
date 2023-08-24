import os
import numpy as np
import pandas as pd
import torch

from config import RESULTS_DIR
from src.training.train_model import train_model
from src.utils import load_synthetic_data, ConsoleProgressBar


def run_experiment(dataset, model_name, weight_sharing, recurrent):
    model_space = {
        "model": model_name,
        "hidden_dim": 64,
        "n_heads": 4,
        "softmax_method": 'gumbel-softmax',
        "n_blocks": 5, "n_layers_per_block": 1, "kernel_size": 2,  # 32
        "dropout": 0.2,
        "recurrent": recurrent,
        "weight_sharing": weight_sharing,
        "aleatoric": True,
        "epistemic": False,
        "uncertainty_contributions": False
    }
    train_space = {
        'lr': 2e-5,
        'epochs': 100,
        'weight_decay': 1e-6,
        'test_size': 0.2,
        "lambda1": 0.0,
        "beta": 0.8,
        "start_coeff": -3,
        "end_coeff": 0,
    }
    unsqueezed = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in dataset.items()}
    return train_model(**unsqueezed, **train_space, **model_space, disable_tqdm=True)


def main():
    data_name = 'synthetic_N-5_T-500_K-6'
    results_path = os.path.join(RESULTS_DIR, "experiments", "complexity", data_name)
    os.makedirs(results_path, exist_ok=True)

    dataset = load_synthetic_data(data_name)

    model_configs = [
        ('NAVAR', False, False),
        ('NAVAR', True, False),
        ('NAVAR', False, True),
        ('NAVAR', True, True),
        ('TAMCaD', False, False),
        ('TAMCaD', True, False),
        ('TAMCaD', False, True),
        ('TAMCaD', True, True),
    ]
    max_evals = 3

    results_list = []

    pbar = ConsoleProgressBar(
        total=max_evals * len(model_configs),
        display_interval=max_evals)

    for model_name, weight_sharing, recurrent in model_configs:
        aucs_best = []
        aucs_last = []
        test_losses = []
        n_params = None

        for _ in range(max_evals):
            result = run_experiment(dataset, model_name, weight_sharing, recurrent)

            test_phase_auc = result['test_phase']['auc']
            aucs_best.append(max(test_phase_auc))
            aucs_last.append(max(test_phase_auc[-10:]))
            test_losses.append(min(result['test_phase']['loss'][-10:]))
            n_params = result['model'].n_params
            pbar.update()

        results_list.append({
            'Model': model_name,
            'Weight Sharing': weight_sharing,
            'Recurrent': recurrent,
            'AUC (best)': np.mean(aucs_best),
            'AUC (final)': np.mean(aucs_last),
            'Loss (final)': np.mean(test_losses),
            'AUC (best) StdDev': np.std(aucs_best) if len(aucs_best) > 2 else None,
            'AUC (final) StdDev': np.std(aucs_last) if len(aucs_last) > 2 else None,
            'Loss (final) StdDev': np.std(test_losses) if len(test_losses) > 2 else None,
            'Params': n_params
        })

    df = pd.DataFrame(results_list)
    csv_file_path = os.path.join(results_path, "results.csv")
    df.to_csv(csv_file_path, index=False)
    print(df)


if __name__ == '__main__':
    main()
