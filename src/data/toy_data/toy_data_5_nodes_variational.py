import torch

from src.data.timeseries_data import TimeSeriesData


def toy_data_5_nodes_variational(batch_size: int = 1, time_steps: int = 500, warmup: int = 200) -> TimeSeriesData:
    """
    Generate a toy dataset with 5 nodes using a variational approach.

    Parameters
    ----------
    batch_size : int, optional (default=1)
        The number of batches to generate.
    time_steps : int, optional (default=500)
        The number of time steps to generate.
    warmup : int, optional (default=200)
        The number of warmup time steps to discard.

    Returns
    -------
    data : TimeSeriesData
        A normalized tensor of size (n_batches, 5_nodes, T), with max_lags=8.
    """
    # Ensure the parameters are valid
    assert batch_size > 0 and time_steps > 0 and warmup >= 0

    # Generate random data
    data = torch.randn((batch_size, 5, warmup + time_steps), dtype=torch.float64) * 0.9

    # Modify data according to a variational approach
    data[:, 4] *= 0.01
    for i in range(1, data.size(-1)):
        data[:, 0, i] += 0.5 * torch.sigmoid(data[:, 3, i - 2])
        data[:, 1, i] += torch.sin(data[:, 0, i - 1])
        data[:, 2, i] += torch.cos(data[:, 1, i - 3].abs())
        factor = min(0.8, (time_steps + warmup - i) / time_steps)
        data[:, 3, i] = 0.3 * data[:, 0, i - 1] + factor * data[:, 1, i - 1]
        data[:, 4, i] += 0.8 * data[:, 4, i - 1] + 0.1 / (0.3 + data[:, 4, i - 2] ** 2)

    # Remove warmup time steps and return as TemporalDataset
    data = data[..., warmup:]
    return TimeSeriesData(name='toy_data_5_nodes_variational', train_data=data, normalize=True)
