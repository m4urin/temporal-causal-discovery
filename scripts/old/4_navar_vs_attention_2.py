import os
from time import sleep

import numpy as np
import torch

from config import RESULTS_DIR
from src.data.generate_toy_data import construct_temporal_causal_data
from src.experiments.train_model import train_model
from src.data.visualisations import plot_multiple_timeseries, plot_heatmap


def run(experiment_name, data, true_causal_matrix):
    model_space = {
        "hidden_dim": 32,
        "n_heads": 4,
        "softmax_method": "softmax",
        "n_blocks": 1, "n_layers_per_block": 2, "kernel_size": 9,  # 17
        "dropout": 0.2,
        "num_variables": data.size(1),
        "lambda1": 0.2,
        "recurrent": False,
        "weight_sharing": False,
        "aleatoric": False,
        "epistemic": False,
        "uncertainty_contributions": False
    }
    train_space = {
        'learning_rate': 5e-4,
        'num_epochs': 2000,
        'weight_decay': 1e-5,
        'test_size': 0.2
    }

    return train_model(train_data=data, true_causal_matrix=true_causal_matrix,
                       **train_space, experiment_name, max_evals=1, causal_data=dataset,
                          model_spaces=model_spaces, train_space=train_space)


if __name__ == '__main__':
    path_causal = os.path.join(RESULTS_DIR, "experiments/4_navar_vs_attention_2/causal")
    # path_random = os.path.join(RESULTS_DIR, "training/4_navar_vs_attention_2/random")
    data_path = os.path.join(RESULTS_DIR, "experiments/4_navar_vs_attention_2/causal_data.pt")
    os.makedirs(path_causal, exist_ok=True)
    # os.makedirs(path_random, exist_ok=True)

    if not os.path.exists(data_path):
        causal_data = construct_temporal_causal_data(num_nodes=6, max_lags=15, sequence_length=1000,
                                                     num_external=2, external_connections=2)
        causal_data.plot('Causal data', view=True,
                         folder_path=os.path.join(RESULTS_DIR, "experiments/4_navar_vs_attention_2"))
        torch.save(causal_data, data_path)
    else:
        causal_data = torch.load(data_path)
        causal_data.plot('Causal data', view=False,
                         folder_path=os.path.join(RESULTS_DIR, "experiments/4_navar_vs_attention_2"))

    causal_results = run("4_navar_vs_attention_2/causal", causal_data.timeseries_data.train_data)  # length=4

    train_losses_causal = [r.train_result.train_losses for r in causal_results]
    train_losses = np.array(train_losses_causal)
    train_losses = train_losses.reshape((1, *train_losses.shape))
    train_losses = smooth_line(train_losses)
    plot_multiple_timeseries(train_losses,
                             labels=["Attention", "Additive"],
                             colors=["red", "blue", "green", "darkorange",
                                     "salmon", "skyblue", "mediumseagreen", "sandybrown"],
                             view=False,
                             limit=(0, 1.25),
                             y_labels=['Train loss'],
                             x_label="Epochs",
                             path=os.path.join(RESULTS_DIR, "experiments/4_navar_vs_attention_2/causal_vs_random.svg")
                             )

    train_losses_true = [interpolate_array(r.train_result.train_losses_true, n=r.train_result.test_every) for r in
                         causal_results]
    train_losses = np.array(train_losses_true)
    train_losses = train_losses.reshape((1, *train_losses.shape))
    train_losses = smooth_line(train_losses)
    plot_multiple_timeseries(train_losses,
                             labels=["Attention", "Additive"],
                             colors=["red", "blue", "green", "darkorange"],
                             view=False,
                             limit=(0, 0.85),
                             y_labels=['Loss'],
                             x_label="Epochs",
                             path=os.path.join(RESULTS_DIR, "experiments/4_navar_vs_attention_2/train_vs_test.svg")
                             )

    aucroc_data_best = [max(result.train_result.aucroc_scores, key=lambda x: x['score']) for result in causal_results]
    aucroc_data_last = [result.train_result.aucroc_scores[-1] for result in causal_results]

    plot_roc_curves(
        fprs=[x['fpr'] for x in aucroc_data_best],
        tprs=[x['tpr'] for x in aucroc_data_best],
        scores=[x['score'] for x in aucroc_data_best],
        names=["Attention", "Additive"],
        view=False,
        path=os.path.join(RESULTS_DIR, "experiments/4_navar_vs_attention_2/aucroc.svg"))

    sleep(0.2)
    for name, x, c in zip(["Attention", "Additive"], aucroc_data_best, causal_results):
        print(f"{name} - score: {round(x['score'], 3)}, epoch: {x['epoch']}")
        print("  ", c.train_config)
        print("  ", c.model_config)
        print()

    size = aucroc_data_best[0]['causal_matrix'].shape
    true_causal_matrix = causal_data.causal_graph.get_causal_matrix(exclude_max_lags=True)[:size[0], :size[1]]
    plot_heatmap(true_causal_matrix,
                 [x['causal_matrix'] for x in aucroc_data_best] +
                 [x['causal_matrix'] for x in aucroc_data_last],
                 view=True,
                 names=["Attention (best)", "Additive (best)", "Attention (last)", "Additive (last)"],
                 path=os.path.join(RESULTS_DIR, "experiments/4_navar_vs_attention_2/causal_matrix.svg"))

    average_prediction = np.array(
        [[x['causal_matrix'] for x in result.train_result.aucroc_scores] for result in causal_results])
    average_prediction = torch.from_numpy(average_prediction)
    average_prediction = average_prediction / average_prediction.sum(dim=-1, keepdim=True)  # normalize
    average_prediction = average_prediction.mean(dim=0)  # (epochs, n, n)
    average_prediction = np.array([[calculate_AUROC(x, true_causal_matrix)['score'] for x in average_prediction]])
    aucroc_scores = np.array([[x['score'] for x in result.train_result.aucroc_scores] for result in causal_results])

    aucroc_scores = np.concatenate((aucroc_scores, average_prediction), axis=0)
    aucroc_scores = np.array([interpolate_array(x, n=causal_results[0].train_result.test_every)
                              for x in aucroc_scores])
    aucroc_scores = smooth_line(aucroc_scores.reshape(1, *aucroc_scores.shape))
    combined_i = aucroc_scores[0, -1].argmax()
    print(f"Combined - score: {round(aucroc_scores[0, -1, combined_i], 3)}, epoch: {combined_i}")

    plot_multiple_timeseries(aucroc_scores,
                             labels=["Attention", "Additive", "Combined"],
                             colors=["red", "blue", "green", "darkorange"],
                             view=True,
                             limit=(0, 1.0),
                             y_labels=['AUCROC'],
                             x_label="Epochs",
                             path=os.path.join(RESULTS_DIR, "experiments/4_navar_vs_attention_2/aucroc_epochs.svg")
                             )

    var = 2
    labels = [f"{i}" for i in range(true_causal_matrix.shape[-1])]
    for i in true_causal_matrix[var].nonzero().flatten():
        labels[i] = labels[i] + " (correct)"
    cms = np.array([[x['causal_matrix'][var] for x in result.train_result.aucroc_scores] for result in causal_results])
    cms = np.transpose(cms, (0, 2, 1))
    plot_multiple_timeseries(cms,
                             labels=labels,
                             view=True,
                             title=f"Variable {var}",
                             y_labels=['Attention', "Additive"],
                             x_label="Epochs",
                             path=os.path.join(RESULTS_DIR, "experiments/4_navar_vs_attention_2/attn_over_time.svg")
                             )
