import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from config import RESULTS_DIR
from src.data.generate_toy_data import construct_temporal_causal_data
from src.data.temporal_causal_data import TemporalCausalData
from src.experiments.train_model import train_model
from src.utils import smooth_line


def run(causal_data: TemporalCausalData, softmax_method, beta, lambda1):
    model_space = {
        "model": "TAMCaD",
        "hidden_dim": 32,
        "n_heads": 4,
        "softmax_method": softmax_method,
        "n_blocks": 1, "n_layers_per_block": 2, "kernel_size": 6,  # 11
        "dropout": 0.2,
        "recurrent": False,
        "weight_sharing": False,
        "aleatoric": True,
        "epistemic": False,
        "uncertainty_contributions": False
    }
    train_space = {
        'lr': 1e-4,
        'epochs': 2000,
        'weight_decay': 1e-6,
        'test_size': 0.2,
        "lambda1": lambda1,
        "beta": beta,
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
    path_causal = os.path.join(RESULTS_DIR, "training/experiment_softmax/causal")
    # path_random = os.path.join(RESULTS_DIR, "training/experiment_softmax/random")
    data_path = os.path.join(RESULTS_DIR, "training/experiment_softmax/causal_data.pt")
    os.makedirs(path_causal, exist_ok=True)
    # os.makedirs(path_random, exist_ok=True)

    if not os.path.exists(data_path):
        causal_data = construct_temporal_causal_data(num_nodes=5, max_lags=10, sequence_length=1252,
                                                     num_external=1, external_connections=1)
        causal_data.plot('Causal data', view=True,
                         folder_path=os.path.join(RESULTS_DIR, "training/experiment_softmax"))
        torch.save(causal_data, data_path)
    else:
        causal_data = torch.load(data_path)
        causal_data.plot('Causal data', view=False,
                         folder_path=os.path.join(RESULTS_DIR, "training/experiment_softmax"))

    gt = causal_data.causal_graph.get_causal_matrix(exclude_max_lags=True, exclude_external_incoming=True)
    #gt = torch.cat((gt[:, :-2].float(), (torch.sum(gt[:, -2:], dim=1, keepdim=True) > 0).float()), dim=1)

    for softmax_method, beta, lambda1 in [('softmax', 0.65, 0.0), ('softmax-1', 0.65, 0.02),
                                          ('normalized-sigmoid', 0.7, 0.1), ('gumbel-softmax', 0.65, 0.0),
                                          ('gumbel-softmax-1', 0.65, 0.02), ('sparsemax', 0.4, 0.0)]:
        test_losses = []
        for i in range(1):
            result, epochs = run(causal_data, softmax_method, beta, lambda1)  # length=4
            plt.plot(range(0, epochs+1, 20), smooth_line(np.array(result['test_phase']['auc']), sigma=1.0), label=softmax_method)

    plt.title('AUROC for various Attention Scoring Methods')
    plt.xlabel('Epochs')
    plt.ylabel('AUROC')
    plt.legend(loc='lower right')
    plt.show()
