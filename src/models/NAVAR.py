import torch
from torch import nn

from src.models.result import Result
from src.models.tcn import TCN
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

"""
class NAVAR_Aleatoric(nn.Module):
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                 dropout=0.0, weight_sharing=False, recurrent=False):
        super().__init__()
        self.n = n_variables

        # TCN to learn additive contributions
        self.contributions = TCN(
            in_channels=3 * n_variables,
            out_channels=n_variables * n_variables,
            hidden_dim=n_variables * hidden_dim,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            groups=n_variables,
            dropout=dropout,
            weight_sharing=weight_sharing,
            recurrent=recurrent
        )

        # TCN to learn aleatoric uncertainty
        self.aleatoric = TCN(
            in_channels=3 * n_variables,
            out_channels=n_variables,
            hidden_dim=2 * hidden_dim,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            groups=1,
            dropout=dropout,
            weight_sharing=weight_sharing,
            recurrent=recurrent
        )

        # Learnable biases
        self.biases = nn.Parameter(torch.ones(1, n_variables, 1) * 0.01)

    def forward(self, x):
        batch_size, _, seq = x.size()

        # Calculate aleatoric uncertainty
        log_var_aleatoric = self.aleatoric(x)  # (batch_size, n, seq)

        # Calculate contributions
        contributions = self.contributions(x).reshape(batch_size, self.n, self.n, seq)

        # Sum contributions and add biases
        prediction = contributions.sum(dim=1) + self.biases

        return {
            'prediction': prediction,
            'contributions': contributions.transpose(1, 2),
            'log_var_aleatoric': log_var_aleatoric
        }

    @staticmethod
    def analysis(prediction, contributions, log_var_aleatoric):
        # Calculate aleatoric and epistemic uncertainties
        # contributions: (bs, n, n, sequence_length)
        aleatoric = torch.exp(0.5 * log_var_aleatoric)  # (bs, n, sequence_length))

        temp_causal_matrix = sliding_window_std(contributions, window=(25, 25), dim=-1)  # (bs, n, n, sequence_length)
        default_causal_matrix = contributions.std(dim=-1)  # (bs, n, n)

        return {
            'prediction': prediction,
            'contributions': contributions,
            'aleatoric': aleatoric,
            'default_causal_matrix': default_causal_matrix,
            'temp_causal_matrix': temp_causal_matrix
        }

    @staticmethod
    def loss_function(y_true, prediction, contributions, log_var_aleatoric, lambda1, coeff):
        aleatoric_loss = NLL_loss(y_true=y_true.unsqueeze(1), y_pred=contributions, log_var=log_var_aleatoric)
        regularization = NAVAR_regularization_loss(contributions, lambda1=lambda1)
        error = ((prediction - y_true) ** 2).mean()
        return error + aleatoric_loss + regularization


class NAVAR_Epistemic(nn.Module):
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                 dropout=0.0, weight_sharing=False, recurrent=False):
        super().__init__()
        self.n = n_variables

        # TCN to learn individual additive contributions
        self.contributions = TCN(
            in_channels=3 * n_variables,
            out_channels=3 * n_variables * n_variables,
            hidden_dim=n_variables * hidden_dim,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            groups=n_variables,
            dropout=dropout,
            weight_sharing=weight_sharing,
            recurrent=recurrent
        )

        # TCN to learn individual uncertainties
        self.uncertainty = TCN(
            in_channels=3 * n_variables,
            out_channels=2 * n_variables,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            groups=1,
            dropout=dropout,
            weight_sharing=weight_sharing,
            recurrent=recurrent
        )

        # Learnable biases
        self.biases = nn.Parameter(torch.ones(1, n_variables, 1) * 0.01)

    def forward(self, x):
        batch_size, _, seq = x.size()

        # Calculate uncertainties for individual contributions
        contr = self.contributions(x).reshape(batch_size, self.n, 3 * self.n, seq)
        contributions, log_v_contr, log_beta_contr = contr.chunk(chunks=3, dim=2)

        log_var_aleatoric_contr = log_beta_contr - log_v_contr
        log_v, log_beta = self.uncertainty(x).chunk(chunks=2, dim=1)
        log_var_aleatoric = log_beta - log_v

        # Calculate contributions and predictions
        prediction = contributions.sum(dim=1) + self.biases

        return {
            'prediction': prediction,
            'contributions': contributions.transpose(1, 2),
            'log_var_aleatoric': log_var_aleatoric,
            'log_v': log_v,
            'log_var_aleatoric_contr': log_var_aleatoric_contr.transpose(1, 2),
            'log_v_contr': log_v_contr.transpose(1, 2)
        }

    @staticmethod
    def analysis(prediction, contributions, log_var_aleatoric, log_v, log_var_aleatoric_contr, log_v_contr):
        # Calculate aleatoric and epistemic uncertainties
        # contributions: (bs, n, n, sequence_length)
        aleatoric_contr = torch.exp(0.5 * log_var_aleatoric_contr)  # (bs, n, n, sequence_length)
        epistemic_contr = torch.exp(-0.5 * log_v_contr)  # (bs, n, n, sequence_length)
        aleatoric = torch.exp(0.5 * log_var_aleatoric)  # (bs, n, sequence_length)
        epistemic = torch.exp(-0.5 * log_v)  # (bs, n, sequence_length)

        temp_confidence_matrix = 1 / epistemic_contr  # (bs, n, n, sequence_length)

        a1 = contributions + 2 * aleatoric_contr
        a2 = contributions - 2 * aleatoric_contr

        temp_c1 = weighted_sliding_window_std(a1, weights=temp_confidence_matrix, window=(25, 25), dim=-1)
        temp_c2 = weighted_sliding_window_std(a2, weights=temp_confidence_matrix, window=(25, 25), dim=-1)
        temp_causal_matrix = (temp_c1 + temp_c2) / 2  # (bs, n, n, sequence_length)

        default_confidence_matrix = temp_confidence_matrix.mean(dim=-1)  # (bs, n, n)

        default_c1 = weighted_std(a1, weights=temp_confidence_matrix, dim=-1)
        default_c2 = weighted_std(a2, weights=temp_confidence_matrix, dim=-1)
        default_causal_matrix = (default_c1 + default_c2) / 2  # (bs, n, n, sequence_length)

        return {
            'prediction': prediction,
            'contributions': contributions,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'aleatoric_contr': aleatoric_contr,
            'epistemic_contr': epistemic_contr,
            'default_causal_matrix': default_causal_matrix,
            'default_confidence_matrix': default_confidence_matrix,
            'temp_causal_matrix': temp_causal_matrix,
            'temp_confidence_matrix': temp_confidence_matrix,
        }

    @staticmethod
    def loss_function(y_true, prediction, contributions, log_var_aleatoric, log_v,
                      log_var_aleatoric_contr, log_v_contr, lambda1, coeff):
        # Calculate epistemic uncertainty term for individual contributions
        epistemic_loss_contr = DER_loss(y_true=y_true.unsqueeze(1), y_pred=contributions,
                                        log_var=log_var_aleatoric_contr, log_v=log_v_contr, coeff=coeff)
        epistemic_loss = DER_loss(y_true=y_true, y_pred=prediction, log_var=log_var_aleatoric, log_v=log_v, coeff=coeff)
        regularization = NAVAR_regularization_loss(contributions, lambda1=lambda1)

        return epistemic_loss_contr + epistemic_loss + regularization

"""
