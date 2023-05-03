from typing import Any
import numpy as np
import torch


class ModelOutput:
    """
    Class to store model output.

    Attributes:
        predictions (Any): model predictions.
        loss (torch.Tensor): model loss.
    """

    def __init__(self, predictions: Any, loss: torch.Tensor):
        self.predictions = predictions
        self.loss = loss

    def __str__(self):
        return f"ModelOutput(predictions={self.predictions}, loss={self.loss})"


class CausalMatrix:
    """
    Class to represent causal matrices.

    Attributes:
        mu (numpy.ndarray): matrix mean.
        std (numpy.ndarray): matrix standard deviation.
        is_history_dependent (bool): whether or not matrix is history-dependent.
        has_std (bool): whether or not matrix has standard deviation.
    """

    def __init__(self, mu, std=None):
        if isinstance(mu, torch.Tensor):
            mu = mu.detach().cpu()

        self.mu = np.array(mu)

        if std is not None:
            if isinstance(std, torch.Tensor):
                std = std.detach().cpu()
            std = np.array(std)
            assert std.shape == self.mu.shape

        self.std = std
        assert self.mu.shape[0] == self.mu.shape[1]
        self.is_history_dependent = len(self.mu.shape) > 2
        self.has_std = self.std is not None

    def __str__(self):
        std_str = f", std={self.std}" if self.has_std else ""
        return f"CausalMatrix(mu={self.mu}{std_str})"


class EvalOutput:
    """
    Class to store evaluation output.

    Attributes:
        predictions (Any): model predictions.
        train_loss (float): training loss.
        test_loss (float): testing loss.
        causal_matrices (dict[str, CausalMatrix]): causal matrices.
    """

    def __init__(self, predictions: Any, train_loss: float, test_loss: float, causal_matrices: dict[str, CausalMatrix]):
        self.predictions = predictions
        self.train_loss = train_loss
        self.test_loss = test_loss
        self.causal_matrices = causal_matrices

    def __str__(self):
        return f"EvalOutput(predictions={self.predictions}, train_loss={self.train_loss}, " \
               f"test_loss={self.test_loss}, causal_matrices={self.causal_matrices})"
