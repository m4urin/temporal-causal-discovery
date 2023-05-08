import os
import zipfile

import numpy as np
import torch

from definitions import DATA_DIR
from src.data.dataset import normalize, Dataset


def get_causeme_data(experiment: str) -> Dataset:
    """
    :return: Dataset object with shape (n_datasets, n_batches, n_nodes, T)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{experiment.split('.zip')[0]}.zip")
    with zipfile.ZipFile(file_path, "r") as f:
        # array with (n_exp, T, n_var)
        data = np.stack([np.loadtxt(f.open(name)) for name in sorted(f.namelist())])
    # (n_exp, T, n_var)
    data = torch.from_numpy(data)
    # standardize (n_exp, T, n_var)
    data = normalize(data, dim=1)
    # create Dataset object
    return Dataset(data.transpose(-1, -2).unsqueeze(dim=1).to(dtype=torch.float32))
