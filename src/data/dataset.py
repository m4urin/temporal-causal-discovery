import bz2
import hashlib
import json
import os
import zipfile
import numpy as np
import pandas as pd
import torch
from typing import Dict, Union

from src.utils import DATA_DIR, OUTPUT_DIR

"""
Structure of the dataset Dictionary:
In the CauseMe project, datasets are represented as dictionaries with the following possible keys:
- 'name' (str): The name of the dataset, which typically includes information about its source or parameters.
- 'data' (torch.Tensor): Represents observation data.
- 'data_noise_adjusted' (torch.Tensor, optional): Represents observations with noise adjustments applied.
- 'ground_truth' (torch.Tensor, optional): Represents the ground truth for comparison and evaluation.
"""


def load_dataset(dataset_type: str, name: str) -> Dict[str, Union[str, torch.Tensor]]:
    """
    Load a dataset from a specified type and name.

    Args:
        dataset_type (str): Type of dataset ('causeme', 'synthetic', or 'dream3').
        name (str): Name of the dataset.

    Returns:
        dict: A dictionary containing dataset information.

    Raises:
        ValueError: If an invalid dataset type is provided.

    """
    dataset_path = os.path.join(DATA_DIR, dataset_type, f'{name}.pt')

    if os.path.exists(dataset_path):
        return torch.load(dataset_path)

    if dataset_type == 'causeme':
        dataset = load_causeme_dataset(name)
    elif dataset_type == 'synthetic':
        dataset = load_synthetic_dataset(name)
    elif dataset_type == 'dream3':
        dataset = load_dream3_dataset(name)
    else:
        raise ValueError(f"Cannot load '{dataset_type}'. Choose from: causeme, dream3, synthetic")

    dataset = normalize_dataset(dataset)
    torch.save(dataset, dataset_path)
    return dataset


def load_synthetic_dataset(name: str) -> Dict[str, Union[str, torch.Tensor]]:
    """
    Load a synthetic dataset from a ZIP file.

    Args:
        name (str): Name of the dataset.

    Returns:
        dict: A dictionary containing synthetic dataset information.

    """
    with zipfile.ZipFile(os.path.join(DATA_DIR, f'synthetic/{name}.zip'), 'r') as archive:
        with archive.open(f'{name}.pt') as pt_file:
            dataset = torch.load(pt_file)
    dataset['name'] = name
    return dataset


def load_causeme_dataset(name: str) -> Dict[str, Union[str, torch.Tensor]]:
    """
    Load a CauseMe dataset from a ZIP file.

    Args:
        name (str): Name of the dataset.

    Returns:
        dict: A dictionary containing CauseMe dataset information.

    """
    with zipfile.ZipFile(os.path.join(DATA_DIR, 'causeme', f"{name}.zip"), 'r') as archive:
        data = np.stack([np.loadtxt(archive.open(name)) for name in sorted(archive.namelist())])
    data = torch.from_numpy(data).float().transpose(-1, -2).unsqueeze(dim=1)
    return {'name': name, 'data': data}


def load_dream3_dataset(name: str) -> Dict[str, Union[str, torch.Tensor]]:
    """
    Load a DREAM3 dataset from a ZIP file.

    Args:
        name (str): Name of the dataset.

    Returns:
        dict: A dictionary containing DREAM3 dataset information.

    """
    with zipfile.ZipFile(os.path.join(DATA_DIR, f'dream3/{name}.zip'), 'r') as archive:
        with archive.open(f'{name}.tsv') as tsv_file:
            df = pd.read_csv(tsv_file, sep='\t', dtype='float32')
        with archive.open(f'{name}_gt.txt') as txt_file:
            txt_lines = [line.decode('utf-8').strip() for line in txt_file]

    data = torch.tensor(df.values, dtype=torch.float32).reshape(-1, 21, 101).transpose(-1, -2)
    data = data[:, 1:]  # Remove time column

    ground_truth = torch.zeros((100, 100), dtype=torch.float32)
    # Read the groundtruth data from the file
    for line in txt_lines:
        source_node, target_node, value = line.strip().split('\t')
        source_node = int(source_node[1:]) - 1  # Remove the 'G' prefix and convert to an integer
        target_node = int(target_node[1:]) - 1  # Remove the 'G' prefix and convert to an integer
        value = float(value)
        # Update the matrix with the groundtruth value
        ground_truth[source_node, target_node] = value
    ground_truth = ground_truth.unsqueeze(0)  # Add batch dimension

    return {'name': name, 'data': data, 'ground_truth': ground_truth}


def normalize_dataset(dataset: Dict[str, Union[str, torch.Tensor]]) -> Dict[str, Union[str, torch.Tensor]]:
    """
    Normalize the dataset.

    Args:
        dataset (dict): A dictionary containing dataset information.

    Returns:
        dict: A dictionary containing normalized dataset information.

    """
    data_mean = dataset['data'].mean(dim=-1, keepdim=True)
    data_std = dataset['data'].std(dim=-1, keepdim=True)

    normalized_dataset = {
        'name': dataset['name'],
        'data': (dataset['data'] - data_mean) / data_std,
        'ground_truth': dataset['ground_truth']
    }
    if 'data_noise_adjusted' in dataset:
        # Use the same mean and std
        normalized_dataset['data_noise_adjusted'] = (dataset['data_noise_adjusted'] - data_mean) / data_std

    return normalized_dataset


def save_synthetic_dataset(name: str, dataset: Dict[str, Union[str, torch.Tensor]]):
    """
    Save a synthetic dataset to a ZIP file as a .pt file.

    Args:
        name (str): Name of the dataset.
        dataset (dict): A dictionary containing synthetic dataset information.

    """
    if 'name' in dataset:
        del dataset['name']

    # Create a directory for the synthetic datasets if it doesn't exist
    synthetic_dir = os.path.join(DATA_DIR, 'synthetic')
    os.makedirs(synthetic_dir, exist_ok=True)

    # Create a ZIP file for the dataset
    with zipfile.ZipFile(os.path.join(synthetic_dir, f'{name}.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        # Save the dataset as a .pt file within the ZIP archive
        with archive.open(f'{name}.pt', 'w') as pt_file:
            torch.save(dataset, pt_file)


def save_causeme_predictions(model_name: str, dataset: Dict[str, torch.Tensor], scores: torch.Tensor, parameters: Dict[str, int], method_sha: str):
    """
    Save predicted ground truths for a CauseMe experiment, ready to be uploaded to the website.

    Args:
        model_name (str): Name of the model used.
        dataset (dict): A dictionary containing dataset information.
        scores (torch.Tensor): Predicted scores.
        parameters (dict): Experiment parameters.
        method_sha (str): SHA identifier for the method.

    """
    # Map method_sha to standard identifiers if needed
    if method_sha == 'NAVAR':
        method_sha = "e0ff32f63eca4587b49a644db871b9a3"
    if method_sha == 'TAMCaD':
        method_sha = "8fbf8af651eb4be7a3c25caeb267928a"

        # Create a data dictionary
    data = {
        'experiment': dataset['name'],
        'model': dataset['name'].split('_')[0],
        'parameter_values': ','.join([f'{k}={v}' for k, v in parameters.items()]),
        'method_sha': method_sha,
        'scores': scores.reshape(scores.size(0), -1).detach().cpu().numpy().tolist()
    }

    # Serialize data and calculate MD5 checksum
    data = json.dumps(data).encode('latin1')
    md5 = hashlib.md5(data).digest().hex()[:8]

    # Create the directory if it doesn't exist
    dir_path = os.path.join(OUTPUT_DIR, 'causeme_results')
    os.makedirs(dir_path, exist_ok=True)

    # Save the data as a compressed JSON file
    file_path = os.path.join(dir_path, f'{model_name}_{dataset["name"]}_{md5}.json.bz2')
    with bz2.BZ2File(file_path, 'w') as bz2_file:
        bz2_file.write(data)

    print('CauseMe predictions written to:', file_path)


if __name__ == '__main__':
    data_frame = load_dataset('synthetic', 'sample_dataset')
    for _k, _v in data_frame.items():
        print(f"{_k}: {_v.shape if isinstance(_v, torch.Tensor) else _v}")
