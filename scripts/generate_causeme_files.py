import os
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from src.utils import load_causeme_data, read_json, get_method_hp_description, write_bz2_file


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


p = "../results/causeme/NAVAR_synthetic_N-5_T-500_K-6"  # Replace this with the desired path

all_folder_names = [f for f in os.listdir(p) if 'synthetic' not in f and os.path.isdir(os.path.join(p, f))]
for folder_name in tqdm(all_folder_names):
    folder_path = os.path.join(p, folder_name)

    splits = folder_name.split('_')
    model_desc = splits[0]
    dataset_name = '_'.join(splits[1:])

    results = torch.load(os.path.join(folder_path, 'results.pt'))
    train_params = read_json(os.path.join(folder_path, 'best_train_params.json'))
    #pretty_print_dict(results[0])
    file_dict = get_method_hp_description(**train_params)
    file_dict['experiment'] = dataset_name
    file_dict['model'] = dataset_name.split('_')[0]

    all_scores = []
    for i, r in enumerate(results):
        temp_causal_matrix = r['train_phase']['causal_matrix_end']  # (1, N, N, T)

        #min_ = torch.nan_to_num(temp_causal_matrix, nan=10000).min()
        #max_ = torch.nan_to_num(temp_causal_matrix, nan=-10000).max()
        #temp_causal_matrix = torch.nan_to_num(temp_causal_matrix, nan=min_, neginf=min_, posinf=max_)
        #temp_causal_matrix = temp_causal_matrix - min_
        #if max_ > 0:
        #    temp_causal_matrix = temp_causal_matrix / max_

        n_nans = temp_causal_matrix.isnan().sum()
        if n_nans > 0:
            #print(min_, max_)
            print(folder_name, "nans:", n_nans)
            assert False
        _max = temp_causal_matrix.max()

        if _max <= 0:
            print(folder_name, 'max:', _max)

        temp_causal_matrix = temp_causal_matrix.mean(dim=(0, -1))  # (N, N)
        if i == 0:
            print(temp_causal_matrix.max())

        temp_causal_matrix = temp_causal_matrix - temp_causal_matrix.min()
        temp_causal_matrix = temp_causal_matrix / temp_causal_matrix.max()

        temp_causal_matrix = temp_causal_matrix.t()  # scores must be A_ij where i causes j
        temp_causal_matrix = temp_causal_matrix.flatten().cpu().numpy()
        all_scores.append(temp_causal_matrix)

    all_scores = np.array(all_scores)
    #all_scores[np.isnan(all_scores)] = 0
    file_dict['scores'] = all_scores.tolist()

    write_bz2_file(os.path.join(folder_path, f'causeme-{folder_name}.json.bz2'), file_dict)

"""
    if 'nonlinear-VAR_N-3_T-3001111111111111' in dataset_name and 'TAMCaD' in model_desc:
        auc = results['train_phase']['end_score']
        auc_best = results['train_phase']['end_score']
        loss = min(results['train_phase']['loss'][-5:])
        print(f"model: {model_desc}, dataset: {dataset_name}, AUC: {auc:.2f}, AUC_best: {auc_best:.2f}, "
              f"loss: {loss:.2f}, n_params: {results['n_parameters_model']}")

    #dataset = load_synthetic_data(dataset_name)

    #plot_heatmap(dataset['gt'][0, 0], [results['train_phase']['causal_matrix_end']], view=True, path=None, names=None)

    #break
"""