import torch

from src.data.causal_graph import TemporalCausalGraph
from src.data.temporal_causal_data import TemporalCausalData
from src.data.timeseries_data import TimeSeriesData


def toy_data_6_nodes_non_additive(batch_size: int = 1, time_steps: int = 500, warmup: int = 200) -> TemporalCausalData:
    """
    Generate a normalized Tensor of size (batch_size, 6, time_steps) with max_lags=13.

    Args:
        batch_size (int): The number of batches. Default is 1.
        time_steps (int): The number of time steps. Default is 500.
        warmup (int): The number of warmup time steps. Default is 200.

    Returns:
        A TemporalCausalData object representing the generated data.
    """
    assert batch_size > 0 and time_steps > 0 and warmup >= 0, \
        "Invalid arguments: batch_size, time_steps, and warmup must be positive integers"

    num_var = 6

    # Generate random data
    data = torch.randn((batch_size, num_var, warmup + time_steps), dtype=torch.float64)
    true_data = torch.zeros_like(data)
    std_data = torch.ones_like(data)

    data *= std_data

    # Scale node 5
    data[:, 5] *= 0.1

    graph = TemporalCausalGraph(num_var, max_lags=13)
    graph.add(0, 1, 2)
    graph.add(1, 0, 2)
    graph.add(1, 4, 10)
    graph.add(2, 5, 0)
    graph.add(2, 4, 3)
    graph.add(3, 2, 5)
    graph.add(3, 0, 4)
    graph.add(4, 2, 3)
    graph.add(4, 3, 12)
    graph.add(5, 5, 0)
    graph.add(5, 5, 1)

    contr = torch.zeros((num_var, num_var, warmup + time_steps), dtype=torch.float64)

    # Compute the time series data
    for i in range(1, data.size()[-1]):
        contr[0, 1, i] = 5 * torch.sigmoid(0.8 * data[:, 1, i - 3])
        true_data[:, 0, i] = contr[0, 1, i]
        data[:, 0, i] += true_data[:, 0, i]

        contr[1, 0, i] = data[:, 0, i - 3] - 1.5
        contr[1, 4, i] = data[:, 4, i - 11] + 0.5
        true_data[:, 1, i] = contr[1, 0, i] * contr[1, 4, i]
        data[:, 1, i] += true_data[:, 1, i]

        contr[2, 4, i] = torch.tanh(data[:, 4, i - 4]) + 0.5
        contr[2, 5, i] = 2 * data[:, 5, i - 1].abs()
        true_data[:, 2, i] = contr[2, 4, i] * contr[2, 5, i]
        data[:, 2, i] += true_data[:, 2, i]
        #print(100 * data[:, 5, i - 1].abs() - 66.8, torch.tanh(data[:, 4, i - 4]) + 0.5, true_data[:, 2, i])

        contr[3, 0, i] = 0.3 * data[:, 0, i - 5]
        contr[3, 2, i] = torch.sin(2 * data[:, 2, i - 6])
        true_data[:, 3, i] = contr[3, 0, i] + contr[3, 2, i]
        data[:, 3, i] += true_data[:, 3, i]

        contr[4, 2, i] = 0.5 * data[:, 2, i - 4]
        contr[4, 3, i] = data[:, 3, i - 13]
        true_data[:, 4, i] = 2 * torch.tanh(1 * (contr[4, 2, i] - contr[4, 3, i]))
        data[:, 4, i] += true_data[:, 4, i]

        contr[5, 5, i] = 0.6 * data[:, 5, i - 1] + 0.28111 / (data[:, 5, i - 3] ** 2 + 0.3)
        true_data[:, 5, i] = contr[5, 5, i]
        #print(true_data[:, 5, i])
        data[:, 5, i] += true_data[:, 5, i]

    # Return only the time series data after warmup period
    data = data[..., warmup:]
    true_data = true_data[..., warmup:]
    std_data = std_data[..., warmup:]
    contr = contr[..., warmup:].float()

    torch.set_printoptions(precision=2, sci_mode=False)
    print('mean')
    print(contr.mean(dim=-1))
    print('std')
    print(contr.std(dim=-1))

    # Return the dataset as a TemporalDataset object
    timeseries_data = TimeSeriesData(train_data=data, true_data=true_data,
                                     std_data=std_data, normalize=True)

    return TemporalCausalData('non_additive', graph, timeseries_data)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    causal_data = toy_data_6_nodes_non_additive()

    causal_data.plot('Toy data non-additive', view=True)
