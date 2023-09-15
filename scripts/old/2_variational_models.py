import os

import numpy as np
import torch

from config import OUTPUT_DIR
from src.data.toy_data.toy_data_6_nodes_non_additive import toy_data_6_nodes_non_additive
from src.experiments.run_experiment import run_experiment
from tests.old.temporal_causal_model import TemporalCausalModel
from src.utils.pytorch import interpolate_array
from src.utils.visualisations import plot_multiple_timeseries, smooth_line


def run(experiment_name, dataset):
    model_space = {
        "model_type": TemporalCausalModel,
        "hidden_dim": 48,
        "kernel_size": 3,
        "n_blocks": 4,
        "n_layers_per_block": 2,
        "dropout": 0.2,
        "num_variables": dataset.causal_graph.num_internal_nodes,
        "max_sequence_length": len(dataset.timeseries_data),
        "lambda1": 0.1,
        "beta1": 0.1,
        "use_navar": True
    }
    train_space = {
        'learning_rate': 4e-4,
        'num_epochs': 3000,
        'optimizer': torch.optim.AdamW,
        'weight_decay': 1e-6,
        'test_size': 0.2
    }

    test = [
        ("TCN", False, False, False),
        ("WS-Rec-TCN", False, True, True),
        ("Var-TCN", True, False, False),
        ("Var-WS-Rec-TCN", True, True, True)
    ]

    model_spaces = [{"name": name, "use_recurrent": rec, "use_weight_sharing": ws,
                     "use_variational_layer": var, **model_space}
                    for name, var, rec, ws in test]

    return run_experiment(experiment_name, max_evals=1, causal_data=dataset,
                          model_spaces=model_spaces, train_space=train_space)


if __name__ == '__main__':
    path = os.path.join(OUTPUT_DIR, "experiments/2_variational_models")
    os.makedirs(path, exist_ok=True)

    causal_data = toy_data_6_nodes_non_additive(time_steps=1000)
    causal_results = run("2_variational_models", causal_data)  # length=4

    train_losses = [interpolate_array(r.train_result.train_losses_true, n=r.train_result.test_every) for r in causal_results]
    train_losses = np.array(train_losses)
    train_losses = train_losses.reshape((1, *train_losses.shape))
    train_losses = smooth_line(train_losses)
    plot_multiple_timeseries(train_losses,
                             labels=["TCN", "WS-Rec-TCN", "Var-TCN", "Var-WS-Rec-TCN"],
                             colors=["red", "blue", "green", "darkorange"],
                             view=False,
                             limit=(0, 0.7),
                             y_labels=['Test loss (regression)'],
                             x_label="Epochs",
                             path=os.path.join(path, "variational.svg")
                             )

