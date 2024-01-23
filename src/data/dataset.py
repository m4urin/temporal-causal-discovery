import os
import zipfile
import numpy as np
import pandas as pd
import torch

from environment import DATA_DIR


def load_dataset(dataset_type: str, name: str):
    dataset_path = os.path.join(DATA_DIR, dataset_type, f'{name}.pt')
    if os.path.exists(dataset_path):
        return torch.load(dataset_path)

    if dataset_type == 'causeme':
        dataset = load_causeme_dataset(name)
    elif dataset_type == 'synthetic':
        dataset = load_synthetic_dataset(name)
    elif dataset_type == 'dream3':
        dataset = load_dream3_data(name)
    else:
        raise ValueError(f"Cannot load '{dataset_type}'. Choose from: causeme, dream3, synthetic")

    dataset = normalize_dataset(dataset)
    torch.save(dataset, dataset_path)
    return dataset


def load_synthetic_dataset(name):
    with zipfile.ZipFile(os.path.join(DATA_DIR, f'synthetic/{name}.zip'), 'r') as archive:
        with archive.open(f'{name}.pt') as pt_file:
            dataset = torch.load(pt_file)
    dataset['name'] = name
    return dataset


def load_causeme_dataset(name):
    with zipfile.ZipFile(os.path.join(DATA_DIR, 'causeme', f"{name}.zip"), 'r') as archive:
        data = np.stack([np.loadtxt(archive.open(name)) for name in sorted(archive.namelist())])
    data = torch.from_numpy(data).float().transpose(-1, -2).unsqueeze(dim=1)
    return {'name': name, 'data': data}


def load_dream3_data(name):
    with zipfile.ZipFile(os.path.join(DATA_DIR, f'dream3/{name}.zip'), 'r') as archive:
        # Read the TSV file into a Pandas DataFrame
        with archive.open(f'{name}.tsv') as tsv_file:
            df = pd.read_csv(tsv_file, sep='\t', dtype='float32')
        with archive.open(f'{name}_gt.txt') as txt_file:
            txt_lines = [line.decode('utf-8').strip() for line in txt_file]

    data = torch.tensor(df.values, dtype=torch.float32).reshape(-1, 21, 101).transpose(-1, -2)
    data = data[:, 1:]  # remove time column

    ground_truth = torch.zeros((100, 100), dtype=torch.float32)
    # Read the groundtruth data from the file
    for line in txt_lines:
        source_node, target_node, value = line.strip().split('\t')
        source_node = int(source_node[1:]) - 1  # Remove the 'G' prefix and convert to an integer
        target_node = int(target_node[1:]) - 1  # Remove the 'G' prefix and convert to an integer
        value = float(value)
        # Update the matrix with the groundtruth value
        ground_truth[source_node, target_node] = value
    ground_truth = ground_truth.unsqueeze(0)  # add batch dimension

    return {'name': name, 'data': data, 'ground_truth': ground_truth}


def normalize_dataset(dataset: dict) -> dict:
    data_mean = dataset['data'].mean(dim=-1, keepdim=True)
    data_std = dataset['data'].std(dim=-1, keepdim=True)

    normalized_dataset = {
        'name': dataset['name'],
        'data': (dataset['data'] - data_mean) / data_std,
        'ground_truth': dataset['ground_truth']
    }
    if 'data_noise_adjusted' in dataset:
        # use the same mean and std
        normalized_dataset['data_noise_adjusted'] = (dataset['data_noise_adjusted'] - data_mean) / data_std

    return normalized_dataset


if __name__ == '__main__':
    data_frame = load_dataset('synthetic', 'synthetic_N-5_T-500_K-6')
    for k, v in data_frame.items():
        print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
