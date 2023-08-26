import os
from pprint import pprint

import torch

from src.data.visualisations import plot_heatmap
from src.utils import load_synthetic_data


def pretty_print_dict(d: dict):
    def tensor_to_size(sub_d: dict):
        result = {}
        for k, v in sub_d.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.size()
            elif isinstance(v, dict):
                result[k] = tensor_to_size(v)
            elif isinstance(v, list):
                result[k] = f"list({len(v)})"
            else:
                result[k] = v
        return result
    pprint(tensor_to_size(d))


p = "../results/gpulab/"  # Replace this with the desired path


for folder_name in [f for f in os.listdir(p) if os.path.isdir(os.path.join(p, f))]:
    folder_path = os.path.join(p, folder_name)
    splits = folder_name.split('_')
    model_desc = splits[0]
    dataset_name = '_'.join(splits[1:])

    if not dataset_name.startswith('synthetic'):
        continue

    results = torch.load(os.path.join(folder_path, 'results.pt'))[0]
    #pretty_print_dict(results)

    #if model_desc in ['NAVAR', 'NAVAR-A', 'TAMCaD-softmax', 'TAMCaD-A-softmax'] and dataset_name == 'synthetic_N-8_T-1000_K-30':
    if 'nonlinear-VAR_N-3_T-300' in folder_name and 'TAMCaD' in folder_name:
        auc = results['train_phase']['end_score']
        auc_best = results['train_phase']['end_score']
        loss = min(results['train_phase']['loss'][-5:])
        print(f"model: {model_desc}, dataset: {dataset_name}, AUC: {auc:.2f}, AUC_best: {auc_best:.2f}, "
              f"loss: {loss:.2f}, n_params: {results['n_parameters_model']}")

    #dataset = load_synthetic_data(dataset_name)

    #plot_heatmap(dataset['gt'][0, 0], [results['train_phase']['causal_matrix_end']], view=True, path=None, names=None)

    #break
