import os
import zipfile

import numpy as np

from definitions import DATA_DIR
from src.data.timeseries_data import TimeSeriesData


def get_causeme_data(experiment_name: str) -> TimeSeriesData:
    """
    Loads the CauseMe dataset from a zip file.

    :param experiment_name: The name of the experiment.
    :return: A TemporalDataset instance.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{experiment_name.split('.zip')[0]}.zip")
    with zipfile.ZipFile(file_path, 'r') as f:
        data = np.stack([np.loadtxt(f.open(name)) for name in sorted(f.namelist())])
    data = data.transpose((0, 2, 1))
    data = np.expand_dims(data, axis=1)
    return TimeSeriesData(name=experiment_name, train_data=data, normalize=True)
