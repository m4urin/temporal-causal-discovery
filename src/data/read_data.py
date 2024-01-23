import os
import zipfile

import pandas as pd
import torch

from environment import DATA_DIR


def get_dream3_data(name):
    dataset_path = os.path.join(DATA_DIR, f'dream3/{name}.pt')
    if os.path.exists(dataset_path):
        return torch.load(dataset_path)

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

    dataset = {'name': name, 'data': data, 'ground_truth': ground_truth}
    torch.save(dataset, dataset_path)

    return dataset


if __name__ == '__main__':
    data_frame = get_dream3_data('yeast3')
    for k, v in data_frame.items():
        print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
