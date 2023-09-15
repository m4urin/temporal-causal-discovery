import torch
from torch import nn

from src.training.result import Result
from src.models.modules.TCN import TCN
from src.utils import sliding_window_std


class NAVAR(nn.Module):
    """
    Implements the Neural Additive Vector Auto-Regression (NAVAR) model.

    Args:
        n_variables (int): Number of variables in the time series.
        hidden_dim (int): Hidden dimension size for the model.
        lambda1 (float, optional): Regularization factor.
        **kwargs: Additional keyword arguments for the underlying :class:`src.models.tcn.TCN` architecture.

    Methods:
        forward(x): Forward pass through the NAVAR model.
        loss_function(y_true, prediction, contributions): Compute the loss for training.
        analysis(prediction, contributions): Perform analysis on the results.
    """
    def __init__(self, n_datasets, n_samples, n_variables, hidden_dim, lambda1, **kwargs):
        super().__init__()
        self.lambda1 = lambda1

        groups = n_datasets * n_samples * n_variables
        self.groups = groups

        # Initialize the TCN model to learn additive contributions.
        self.contributions = TCN(
            in_channels=groups * 3,
            out_channels=groups * n_variables,
            hidden_dim=groups * hidden_dim,
            groups=groups,
            **kwargs
        )

        # Initialize learnable biases with small values.
        self.biases = nn.Parameter(torch.ones(1, n_datasets, n_samples, n_variables, 1) * 0.01)

    def forward(self, x):
        """
        Forward pass through the NAVAR model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n, sequence_length).

        Returns:
            dict: Dictionary containing the prediction and contributions.
        """
        # batch_size, n_datasets, n_samples, n_var, seq
        batch_size, n_datasets, n_samples, n, seq = x.size()

        x = x.reshape(batch_size, -1, seq)

        # Calculate contributions and reshape it.
        contributions = self.contributions(x)
        contributions = contributions.reshape(batch_size, n_datasets, n_samples, n, n, seq)

        # Sum contributions along dimension 1 and add biases to get the final prediction.
        prediction = contributions.sum(dim=-3) + self.biases

        return {
            'prediction': prediction,
            'contributions': contributions.transpose(-2, -3)
        }

    def loss_function(self, y_true, prediction, contributions):
        """
        Compute the loss for training the model.

        Args:
            y_true (torch.Tensor): Ground truth labels of size (batch_size, n_datasets, n, seq).
            prediction (torch.Tensor): Model predictions.
            contributions (torch.Tensor): Contributions calculated by the model.

        Returns:
            torch.Tensor: Computed loss.
        """
        # Mean squared error loss
        loss = nn.functional.mse_loss(prediction, y_true.unsqueeze(2))

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
