import os

import numpy as np
import torch
from matplotlib import pyplot as plt
import pprint

from definitions import RESULTS_DIR
from src.data.generate_toy_data import construct_temporal_causal_data
from src.data.temporal_causal_data import TemporalCausalData
from src.experiments.train_model import train_model
from src.utils import tensor_dict_to_str, smooth_line
from src.visualisations import plot_heatmap


def run(causal_data: TemporalCausalData, model_name, weight_sharing, recurrent):
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
        'epochs': 8000,
        'weight_decay': 1e-6,
        'test_size': 0.2,
        "lambda1": 0.0,
        "beta": 0.8,
        "start_coeff": -3,
        "end_coeff": 0,
    }
    train_data = causal_data.timeseries_data.train_data
    true_causal_matrix = causal_data.causal_graph.get_causal_matrix(exclude_max_lags=True,
                                                                    exclude_external_incoming=True,
                                                                    exclude_external_outgoing=True)
    true_causal_matrix = true_causal_matrix[None, :, :, None].expand(-1, -1, -1, train_data.size(-1))
    train_data = causal_data.timeseries_data.train_data

    result = train_model(train_data=train_data, true_causal_matrix=true_causal_matrix, **train_space, **model_space)
    return result, train_space['epochs']


if __name__ == '__main__':
    path_causal = os.path.join(RESULTS_DIR, "experiments/experiment_complexity/causal")
    data_path = os.path.join(RESULTS_DIR, "experiments/experiment_complexity/causal_data.pt")
    os.makedirs(path_causal, exist_ok=True)
    # os.makedirs(path_random, exist_ok=True)

    if not os.path.exists(data_path):
        causal_data = construct_temporal_causal_data(num_nodes=5, max_lags=30, sequence_length=1252,
                                                     num_external=1, external_connections=1)
        causal_data.plot('Causal data', view=True,
                         folder_path=os.path.join(RESULTS_DIR, "experiments/experiment_complexity"))
        torch.save(causal_data, data_path)
    else:
        causal_data = torch.load(data_path)
        causal_data.plot('Causal data', view=False,
                         folder_path=os.path.join(RESULTS_DIR, "experiments/experiment_complexity"))

    gt = causal_data.causal_graph.get_causal_matrix(exclude_max_lags=True, exclude_external_incoming=True)

    plots = False
    epochs = None
    all_aucs = []

    for model_name in ['NAVAR', 'TAMCaD']:
        for weight_sharing in [False, True]:
            for recurrent in [False, True]:
                aucs_best = []
                aucs_last = []
                test_losses = []
                n_params = None
                for i in range(1):
                    result, epochs = run(causal_data, model_name, weight_sharing, recurrent)  # length=4

                    aucs_best.append(max(result['test_phase']['auc']))
                    aucs_last.append(max(result['test_phase']['auc'][-10:]))
                    test_losses.append(min(result['test_phase']['loss'][-10:]))
                    all_aucs.append(result['test_phase']['auc'])
                    n_params = result['model'].n_params
                    if plots:
                        plt.plot(range(0, epochs+1, 20), result['train_phase']['loss'], label='train')
                        plt.plot(range(0, epochs+1, 20), result['test_phase']['loss'], label='test')
                        plt.legend()
                        plt.show()

                        plt.plot(range(0, epochs+1, 20), result['train_phase']['auc'], label='train')
                        plt.plot(range(0, epochs+1, 20), result['test_phase']['auc'], label='test')
                        plt.legend()
                        plt.show()

                        cm_end = result['train_phase']['causal_matrix_end'].mean(dim=(0, 3)).cpu().numpy()
                        cm_best = result['train_phase']['causal_matrix_best'].mean(dim=(0, 3)).cpu().numpy()
                        if result['train_phase']['ext_matrix_best'] is not None:
                            ext_best = result['train_phase']['ext_matrix_best'].mean(dim=(0, 3)).cpu().numpy()
                            ext_end = result['train_phase']['ext_matrix_best'].mean(dim=(0, 3)).cpu().numpy()
                            cm_end = np.concatenate((cm_end, ext_end), axis=-1)
                            cm_best = np.concatenate((cm_best, ext_best), axis=-1)

                        M = [cm_best, cm_end]

                        if result['train_phase']['conf_matrix_best'] is not None:
                            conf_best = result['train_phase']['conf_matrix_best'].mean(dim=(0, 3)).cpu().numpy()
                            conf_end = result['train_phase']['conf_matrix_best'].mean(dim=(0, 3)).cpu().numpy()
                            M.extend([conf_best, conf_end])

                        plot_heatmap(gt, M, view=True, names=['Causal (best)', 'Causal (last)', 'Confidence (best)', 'Confidence (last)'])

                print(model_name, ', weight_sharing:', weight_sharing, ', recurrent:', recurrent, ', n_params:', n_params)
                print('\tAUC (best)', round(np.mean(aucs_best), 2), '±', round(np.std(aucs_best), 2) if len(aucs_best) > 2 else '_')
                print('\tAUC (final)', round(np.mean(aucs_last), 2), '±', round(np.std(aucs_last), 2) if len(aucs_last) > 2 else '_')
                print('\tLoss (final)', round(np.mean(test_losses), 2), '±', round(np.std(test_losses), 2) if len(test_losses) > 2 else '_')

    i = 0
    for model_name in ['TAMCaD', 'NAVAR']:
        for weight_sharing in [False, True]:
            for recurrent in [False, True]:
                label_str = [model_name]
                if recurrent:
                    label_str.append('Rec')
                if weight_sharing:
                    label_str.append('WS')
                plt.plot(range(0, epochs + 1, 20), smooth_line(np.array(all_aucs[i]), sigma=1.0),
                         label="-".join(label_str))
                i += 1
    plt.title('AUROC for Weight-sharing (WS) and Recurrent layers (Rec)')
    plt.xlabel('Epochs')
    plt.ylabel('AUROC')
    plt.legend(loc='lower right')
    plt.show()

