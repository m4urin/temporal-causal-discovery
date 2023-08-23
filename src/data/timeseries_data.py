from typing import Union

import numpy as np
import torch

from src.visualisations import plot_multiple_timeseries


def verify(x: Union[np.ndarray, torch.Tensor], num_dim=None, shape=None):
    if num_dim is not None:
        assert len(x.shape) == num_dim, f'attribute should have {num_dim} dimensions'
    if shape is not None:
        assert x.shape == shape, f'attribute should be of shape {tuple(shape)}'
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.dtype != torch.float32:
        x = x.float()
    return x


class TimeSeriesData:
    """
    A PyTorch Dataset for handling temporal data.

    Args:
        train_data (torch.Tensor): A 3D tensor of shape (batch_size, num_variables, sequence_length) containing the data.
        normalize (bool, optional): Flag indicating whether to normalize the data.

    Attributes:
        train_data (torch.Tensor): A 3D tensor of shape (batch_size, num_variables, sequence_length) containing the data.
        batch_size (int): The size of the batch.
        num_variables (int): The number of variables in the data.
        sequence_length (int): The length of the sequence.
    """

    def __init__(self,
                 train_data: Union[np.ndarray, torch.Tensor],
                 true_data: Union[np.ndarray, torch.Tensor] = None,
                 std_data: Union[np.ndarray, torch.Tensor] = None,
                 normalize: bool = False) -> None:
        super().__init__()
        self.train_data = verify(train_data, num_dim=3)
        self.batch_size, self.num_variables, self.sequence_length = self.train_data.shape

        self.true_data = None
        if true_data is not None:
            self.true_data = verify(true_data, shape=self.train_data.shape)

        self.std_data = None
        if std_data is not None:
            self.std_data = verify(std_data, shape=self.train_data.shape)

        if normalize:
            self.normalize()

    def cuda(self):
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, torch.Tensor):
                setattr(self, name, attr.cuda())
        return self

    def cpu(self):
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, torch.Tensor):
                setattr(self, name, attr.detach().cpu())
        return self

    def train_test_split(self, test_size: Union[int, float]):
        if isinstance(test_size, int):
            train_size = self.sequence_length - test_size
        elif isinstance(test_size, float):
            train_size = int(self.sequence_length * (1.0 - test_size))
        else:
            raise Exception("TODO")
        return self[..., :train_size], self[..., train_size:]

    def __getitem__(self, index) -> 'TimeSeriesData':
        """
        Gets a single item from the dataset.

        Args:
            index (int): The index of the item to get.

        Returns:
            torch.Tensor: A 2D tensor of shape (num_variables, sequence_length) containing the data for the item.
        """
        true_data = self.true_data[index] if self.true_data is not None else None
        std_data = self.std_data[index] if self.std_data is not None else None
        return TimeSeriesData(self.train_data[index], true_data, std_data)

    def normalize(self):
        """
        Normalizes the data along the sequence axis.
        """
        mu = self.train_data.mean(dim=-1, keepdim=True)
        std = self.train_data.std(dim=-1, keepdim=True) + 1e-8
        self.train_data = (self.train_data - mu) / std
        if self.true_data is not None:
            self.true_data = (self.true_data - mu) / std
        if self.std_data is not None:
            self.std_data = (self.std_data - mu) / std

    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset.

        Returns:
            str: A string representation of the dataset.
        """
        return f"TemporalDataset(" \
               f"batch_size={self.batch_size}, " \
               f"num_variables={self.num_variables}, " \
               f"sequence_length={self.sequence_length}, " \
               f"dtype={self.train_data.dtype})"

    def __str__(self) -> str:
        """
        Returns a string representation of the dataset.

        Returns:
            str: A string representation of the dataset.
        """
        return repr(self)

    def __len__(self):
        return self.train_data.size(-1)

    def plot(self, title=None, path=None, view=False):
        plot_data = torch.stack((
            self.train_data[0, :, -300:],
            self.true_data[0, :, -300:])).permute(1, 0, 2)
        plot_multiple_timeseries(plot_data,
                                 title=title,
                                 y_labels=[f"N{i + 1}" for i in range(self.train_data.size(1))],
                                 labels=['Sampled', 'True mean'],
                                 limit=(-2, 2),
                                 path=path,
                                 view=view)
