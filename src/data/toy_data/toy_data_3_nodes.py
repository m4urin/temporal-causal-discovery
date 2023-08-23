import torch

from src.data.timeseries_data import TimeSeriesData


def toy_data_3_nodes(batch_size: int = 1, time_steps: int = 500, warmup: int = 200) -> TimeSeriesData:
    """
    Generate a dataset of time series with 3 nodes.

    :param batch_size: The number of time series to generate.
    :param time_steps: The length of each time series, including the warmup period.
    :param warmup: The number of initial time steps to discard as warmup.
    :return: A TemporalDataset object with shape (n_batches, 3_nodes, T), max_lags=1.
    """
    assert batch_size > 0 and time_steps > 0 and (
                warmup >= 0), "batch_size, time_steps and warmup must be positive integers"

    # Generate random data of shape (batch_size, 3, warmup + time_steps)
    data = torch.randn((batch_size, 3, warmup + time_steps), dtype=torch.float64)

    # Generate time series data using a recurrence relation for each node
    for i in range(1, data.size(-1)):
        data[:, 0, i] += torch.cos(data[:, 1, i - 1]) + torch.tanh(data[:, 2, i - 1])
        data[:, 1, i] += 0.35 * data[:, 1, i - 1] + data[:, 2, i - 1]
        data[:, 2, i] += torch.abs(0.5 * data[:, 0, i - 1]) + torch.sin(2 * data[:, 1, i - 1])

    # Remove the warmup period from the data
    data = data[..., warmup:]

    # Return the dataset as a TemporalDataset object
    return TimeSeriesData(name="toy_data_3_nodes", train_data=data, normalize=True)
