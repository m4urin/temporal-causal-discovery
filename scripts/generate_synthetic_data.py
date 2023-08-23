import os.path

import torch
from tqdm import trange

from definitions import DATA_DIR, DEVICE
from src.synthetic_data.temporal_causal_data import TemporalCausalData
from src.synthetic_data.causal_graph import SyntheticTCG
from src.synthetic_data.timeseries_data import TimeSeriesData


def construct_random(num_nodes: int = 3, sequence_length: int = 500) -> TemporalCausalData:
    """
    Generates a random time series dataset with the given parameters.
    :param num_nodes: The number of variables in each batch
    :param sequence_length: The length of the time series
    :return: A TemporalDataset object with shape (n_batches, num_variables, sequence_length)
    """
    graph = SyntheticTCG(num_internal_nodes=num_nodes, max_lags=1,
                         causal_matrix=torch.zeros(num_nodes, num_nodes, 1).bool())  # no causal connections
    random_data = torch.randn(1, num_nodes, sequence_length)
    time_series_data = TimeSeriesData(random_data, random_data, torch.zeros_like(random_data), normalize=True)
    return TemporalCausalData(name='random', causal_graph=graph, timeseries_data=time_series_data)


def construct_temporal_causal_data(num_nodes, max_lags, num_external=0, minimum_incoming_connections=2,
                                   external_connections=1, sequence_length: int = 500, warmup: int = 200,
                                   noise_factor: float = 0.5):
    assert sequence_length > 0 and warmup >= 0, \
        "Invalid arguments: time_steps, and warmup must be positive integers"

    total_nodes = num_nodes + num_external
    # Generate random data
    data = torch.randn((total_nodes, max_lags + warmup + sequence_length + 1), dtype=torch.float32, device=DEVICE)
    std_data = torch.ones_like(data, device=DEVICE)
    std_data[:num_nodes] *= noise_factor
    data[:, max_lags:] *= std_data[:, max_lags:]
    true_data = torch.zeros_like(data, device=DEVICE)

    causal_graph = SyntheticTCG(
        num_internal_nodes=num_nodes,
        max_lags=max_lags,
        num_external_nodes=num_external,
        num_ext_connections=external_connections,
        min_incoming_connections=minimum_incoming_connections)
    causal_graph.init_functional_relationships()

    with torch.no_grad():
        for i in trange(data.size(-1) - max_lags - 1, desc='Generate sequence..'):
            x = causal_graph(data[:, i:i + max_lags])
            true_data[:, i + max_lags] = x
            data[:, i + max_lags] += x

    name = f"nodes_{num_nodes}_T_{sequence_length}_lags_{max_lags}_" \
           f"ext_{num_external}_difficulty_{causal_graph.difficulty_score}"

    return TemporalCausalData(
        name=name,
        causal_graph=causal_graph,
        timeseries_data=TimeSeriesData(
            train_data=data[None, :num_nodes, -sequence_length:].cpu(),
            true_data=true_data[None, :num_nodes, -sequence_length:].cpu(),
            std_data=std_data[None, :num_nodes, -sequence_length:].cpu(),
            normalize=True
        )
    )


if __name__ == '__main__':
    causal_dataset = construct_temporal_causal_data(num_nodes=3, max_lags=4, num_external=1, external_connections=2,
                                                    sequence_length=300, minimum_incoming_connections=2,
                                                    noise_factor=0.3)

    folder = os.path.join(DATA_DIR, 'toy_data', causal_dataset.name)
    os.makedirs(folder, exist_ok=True)

    causal_dataset.save(folder)

