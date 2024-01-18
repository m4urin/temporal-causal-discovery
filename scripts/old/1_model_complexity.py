import os

import numpy as np
import torch

from environment import OUTPUT_DIR
from src.data.generate_toy_data import construct_random
from src.data.toy_data.toy_data_6_nodes_non_additive import toy_data_6_nodes_non_additive
from src.eval.soft_auroc import calculate_AUROC
from src.experiments.run_experiment import run_experiment
from tests.old.temporal_causal_model import TemporalCausalModel
from src.utils.pytorch import interpolate_array
from src.utils.visualisations import plot_multiple_timeseries, smooth_line, plot_heatmap, plot_roc_curves


def run(experiment_name, dataset):
    model_space = {
        "model_type": TemporalCausalModel,
        "hidden_dim": 64,
        "kernel_size": 3,
        "n_blocks": 6,
        "n_layers_per_block": 2,
        "dropout": 0.2,
        "num_variables": dataset.causal_graph.num_internal_nodes,
        "max_sequence_length": len(dataset.timeseries_data),
        "lambda1": 0.1,
        "beta1": 0.1,
        "use_navar": True
    }
    train_space = {
        'learning_rate': 1e-3,
        'num_epochs': 3500,
        'optimizer': torch.optim.AdamW,
        'weight_decay': 1e-5,
        'test_size': 0.2
    }

    test = [
        ("TCN", False, False),
        ("Rec-TCN", True, False),
        ("WS-TCN", False, True),
        ("WS-Rec-TCN", True, True)
    ]

    model_spaces = [{"name": name, "use_recurrent": rec, "use_weight_sharing": ws, **model_space}
                    for name, rec, ws in test]

    return run_experiment(experiment_name, max_evals=1, causal_data=dataset,
                          model_spaces=model_spaces, train_space=train_space)


if __name__ == '__main__':
    path_causal = os.path.join(OUTPUT_DIR, "experiments/1_model_complexity/causal")
    path_random = os.path.join(OUTPUT_DIR, "experiments/1_model_complexity/random")
    os.makedirs(path_causal, exist_ok=True)
    os.makedirs(path_random, exist_ok=True)

    causal_data = toy_data_6_nodes_non_additive(time_steps=1500)
    random_data = construct_random(
        num_nodes=causal_data.causal_graph.num_internal_nodes,
        sequence_length=len(causal_data.timeseries_data))

    causal_results = run("1_model_complexity/causal", causal_data)  # length=4
    random_results = run("1_model_complexity/random", random_data)

    train_losses_causal = [r.train_result.train_losses for r in causal_results]
    train_losses_random = [r.train_result.train_losses for r in random_results]
    train_losses = np.array(train_losses_causal + train_losses_random)
    train_losses = train_losses.reshape((1, *train_losses.shape))
    train_losses = smooth_line(train_losses)
    plot_multiple_timeseries(train_losses,
                             labels=["TCN_causal", "Rec-TCN_causal", "WS-TCN_causal", "WS-Rec-TCN_causal",
                                     "TCN_random", "Rec-TCN_random", "WS-TCN_random", "WS-Rec-TCN_random"],
                             colors=["red", "blue", "green", "darkorange",
                                     "salmon", "skyblue", "mediumseagreen", "sandybrown"],
                             view=False,
                             limit=(0, 1.25),
                             y_labels=['Train loss'],
                             x_label="Epochs",
                             path=os.path.join(OUTPUT_DIR, "experiments/1_model_complexity/causal_vs_random.svg")
                             )

    train_losses_true = [interpolate_array(r.train_result.train_losses_true, n=r.train_result.test_every) for r in causal_results]
    train_losses = np.array(train_losses_true)
    train_losses = train_losses.reshape((1, *train_losses.shape))
    train_losses = smooth_line(train_losses)
    plot_multiple_timeseries(train_losses,
                             labels=["TCN", "Rec-TCN", "WS-TCN", "WS-Rec-TCN"],
                             colors=["red", "blue", "green", "darkorange"],
                             view=False,
                             limit=(0, 0.85),
                             y_labels=['Loss'],
                             x_label="Epochs",
                             path=os.path.join(OUTPUT_DIR, "experiments/1_model_complexity/train_vs_test.svg")
                             )

    true_causal_matrix = causal_data.causal_graph.causal_matrix.sum(dim=-1).bool().float().numpy()
    pred_causal_matrix = [r.model_output.attn[0, :, :, 50:].std(dim=-1).numpy() for r in causal_results]

    auroc = [calculate_AUROC(m, true_causal_matrix) for m in pred_causal_matrix]

    plot_roc_curves(
        fprs=[fpr for _, fpr, _, _ in auroc],
        tprs=[tpr for _, _, tpr, _ in auroc],
        scores=[score for score, _, _, _ in auroc],
        names=["TCN", "Rec-TCN", "WS-TCN", "WS-Rec-TCN"],
        view=False,
        path=os.path.join(OUTPUT_DIR, "experiments/1_model_complexity/roc.svg"))
    for n, score in zip(["TCN", "Rec-TCN", "WS-TCN", "WS-Rec-TCN"], auroc):
        print(f"{n}: {round(score[0], 2)}")

    plot_heatmap(true_causal_matrix, pred_causal_matrix, view=True,
                 names=["TCN", "Rec-TCN", "WS-TCN", "WS-Rec-TCN"],
                 path=os.path.join(OUTPUT_DIR, "experiments/1_model_complexity/causal_matrix.svg"))
