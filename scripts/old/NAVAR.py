import torch
from torch import nn

from src.training.result import Result
from src.models.TCN import TCN
from src.utils import sliding_window_std


class NAVAR(nn.Module):
    """
    Implements the Neural Additive Vector Auto-Regression (NAVAR) model.

    Args:
        n_datasets (int): Number of distinct datasets.
        n_ensembles (int): Number of ensembles per dataset.
        n_variables (int): Number of variables in the time series.
        hidden_dim (int): Hidden dimension size for the model.
        lambda1 (float, optional): Regularization factor.
        **kwargs: Additional keyword arguments for the underlying :class:`src.models.tcn.TCN` architecture.

    Methods:
        forward(x): Forward pass through the NAVAR model.
        loss_function(y_true, prediction, contributions): Compute the loss for training.
        analysis(prediction, contributions): Perform analysis on the results.
    """
    def __init__(self, n_datasets, n_ensembles, n_variables, hidden_dim, lambda1, **kwargs):
        super().__init__()
        self.lambda1 = lambda1
        self.groups = n_datasets * n_ensembles * n_variables

        # Initialize the TCN model to learn additive contributions.
        self.contributions = TCN(
            in_channels=self.groups * 3,
            out_channels=self.groups * n_variables,
            hidden_dim=self.groups * hidden_dim,
            groups=self.groups,
            **kwargs
        )

        # Initialize learnable biases with small values.
        self.biases = nn.Parameter(torch.ones(n_datasets, n_ensembles, n_variables, 1) * 0.01)

    def forward(self, x):
        """
        Forward pass through the NAVAR model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_datasets, n_ensembles, n_variables * 3, sequence_length).

        Returns:
            dict: Dictionary containing the prediction and contributions.
        """
        batch_size, n_datasets, n_ensembles, n_var, sequence_length = x.size()
        n_var = n_var // 3

        x = x.reshape(batch_size, -1, sequence_length)

        # Calculate contributions and reshape it.
        contributions = self.contributions(x).reshape(batch_size, n_datasets, n_ensembles, n_var, n_var, -1)

        # Sum contributions along dimension 1 and add biases to get the final prediction.
        prediction = contributions.sum(dim=-3) + self.biases

        return {
            'prediction': prediction,  # (batch_size, n_datasets, n_ensembles, n_var, seq)
            'contributions': contributions.transpose(-2, -3)  # (batch_size, n_datasets, n_ensembles, n_var, n_var, seq)
        }

    def compute_loss(self, y_true, prediction, contributions):
        loss = ((prediction - y_true) ** 2).mean()  # MSE loss

        if self.lambda1:
            # Adding regularization term
            loss = loss + self.lambda1 * contributions.abs().mean()

        return loss

    def analysis(self, prediction, contributions):
        """
        Perform analysis on the results produced by the model.

        Args:
            prediction (torch.Tensor): Model predictions.
            contributions (torch.Tensor): Contributions calculated by the model.

        Returns:
            dict: Dictionary containing analysis results.
        """
        # Return prediction, contributions, default and temporal causal matrices.
        sliding_contributions = sliding_window_std(contributions, window=(30, 30), dim=-1)
        contributions_std = contributions.std(dim=-1)

        return Result(
            prediction=prediction.mean(dim=2),  # (batch_size, n_datasets, n, T)
            causal_matrix=contributions_std.mean(dim=2),  # (batch_size, n_datasets, n, n)
            causal_matrix_std=contributions_std.std(dim=2),  # (batch_size, n_datasets, n, n)
            temporal_causal_matrix=sliding_contributions.mean(dim=2),  # (batch_size, n_datasets, n, n, T)
            temporal_causal_matrix_std=sliding_contributions.std(dim=2)  # (batch_size, n_datasets, n, n, T)
        )
