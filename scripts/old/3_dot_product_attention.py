import os

import numpy as np
import torch

from definitions import RESULTS_DIR
from src.data.generate_toy_data import construct_temporal_causal_data
from src.experiments.run_experiment import run_experiment
from tests.old.temporal_causal_model import TemporalCausalModel
from src.utils.pytorch import interpolate_array
from src.utils.visualisations import plot_multiple_timeseries, smooth_line, plot_heatmap, plot_roc_curves


def run(experiment_name, dataset):
    model_space = {
        "model_type": TemporalCausalModel,
        "hidden_dim": 64,
        "kernel_size": 3,
        "n_blocks": 4,
        "n_layers_per_block": 2,
        "dropout": 0.2,
        "num_variables": dataset.causal_graph.num_nodes - dataset.causal_graph.num_external,
        "max_sequence_length": len(dataset.timeseries_data),
        "lambda1": 0.1,
        "beta1": 0.0,
        "use_attentions": True,
        "use_instantaneous_predictions": False,
        "num_external_variables": dataset.causal_graph.num_external
    }
    train_space = {
        'learning_rate': 1e-4,
        'num_epochs': 3000,
        'optimizer': torch.optim.AdamW,
        'weight_decay': 1e-6,
        'test_size': 0.2
    }
    print('receptive_field',
          (2 ** model_space['n_blocks'] - 1) * model_space["n_layers_per_block"] * (model_space["kernel_size"] - 1) + 1)

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
    path_causal = os.path.join(RESULTS_DIR, "experiments/3_dot_product_attention/causal")
    # path_random = os.path.join(RESULTS_DIR, "experiments/3_dot_product_attention/random")
    data_path = os.path.join(RESULTS_DIR, "experiments/3_dot_product_attention/causal_data.pt")
    os.makedirs(path_causal, exist_ok=True)
    # os.makedirs(path_random, exist_ok=True)

    if not os.path.exists(data_path):
        causal_data = construct_temporal_causal_data(num_nodes=14, num_edges=20, sequence_length=1000,
                                                     max_lags=60, ext_nodes=2, num_ext_connections=1)
        causal_data.plot('Causal data', view=True,
                         folder_path=os.path.join(RESULTS_DIR, "experiments/3_dot_product_attention"))
        torch.save(causal_data, data_path)
    else:
        causal_data = torch.load(data_path)
        causal_data.plot('Causal data', view=True,
                         folder_path=os.path.join(RESULTS_DIR, "experiments/3_dot_product_attention"))

    causal_results = run("3_dot_product_attention/causal", causal_data)  # length=4
    # random_results = run("3_dot_product_attention/random", random_data)

    train_losses_causal = [r.train_result.train_losses for r in causal_results]
    # train_losses_random = [r.train_result.train_losses for r in random_results]
    train_losses = np.array(train_losses_causal)
    # train_losses = np.array(train_losses_causal + train_losses_random)
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
                             path=os.path.join(RESULTS_DIR, "experiments/3_dot_product_attention/causal_vs_random.svg")
                             )

    plot_multiple_timeseries(train_losses,
                             labels=["TCN_causal", "Rec-TCN_causal", "WS-TCN_causal", "WS-Rec-TCN_causal",
                                     "TCN_random", "Rec-TCN_random", "WS-TCN_random", "WS-Rec-TCN_random"],
                             colors=["red", "blue", "green", "darkorange",
                                     "salmon", "skyblue", "mediumseagreen", "sandybrown"],
                             view=False,
                             limit=(0, 1.25),
                             y_labels=['Train loss'],
                             x_label="Epochs",
                             path=os.path.join(RESULTS_DIR, "experiments/3_dot_product_attention/causal_vs_random.svg")
                             )

    train_losses_true = [interpolate_array(r.train_result.train_losses_true, n=r.train_result.test_every) for r in
                         causal_results]
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
                             path=os.path.join(RESULTS_DIR, "experiments/3_dot_product_attention/train_vs_test.svg")
                             )

    aucroc_data = [max(result.train_result.aucroc_scores, key=lambda x: x['score']) for result in causal_results]

    plot_roc_curves(
        fprs=[x['fpr'] for x in aucroc_data],
        tprs=[x['tpr'] for x in aucroc_data],
        scores=[x['score'] for x in aucroc_data],
        names=["TCN", "Rec-TCN", "WS-TCN", "WS-Rec-TCN"],
        view=False,
        path=os.path.join(RESULTS_DIR, "experiments/3_dot_product_attention/aucroc.svg"))

    for name, x in zip(["TCN", "Rec-TCN", "WS-TCN", "WS-Rec-TCN"], aucroc_data):
        print(f"{name} - score: {round(x['score'], 2)}, epoch: {x['epoch']}")

    true_causal_matrix = causal_data.causal_graph.get_causal_matrix()
    plot_heatmap(true_causal_matrix, [x['causal_matrix'] for x in aucroc_data], view=True,
                 names=["TCN", "Rec-TCN", "WS-TCN", "WS-Rec-TCN"],
                 path=os.path.join(RESULTS_DIR, "experiments/3_dot_product_attention/causal_matrix.svg"))

    aucroc_scores = np.array([interpolate_array([x['score'] for x in result.train_result.aucroc_scores],
                                                n=result.train_result.test_every) for result in causal_results])
    aucroc_scores = smooth_line(aucroc_scores.reshape(1, *aucroc_scores.shape))

    plot_multiple_timeseries(aucroc_scores,
                             labels=["TCN", "Rec-TCN", "WS-TCN", "WS-Rec-TCN"],
                             colors=["red", "blue", "green", "darkorange"],
                             view=True,
                             limit=(0, 1.0),
                             y_labels=['AUCROC'],
                             x_label="Epochs",
                             path=os.path.join(RESULTS_DIR, "experiments/3_dot_product_attention/aucroc_epochs.svg")
                             )
