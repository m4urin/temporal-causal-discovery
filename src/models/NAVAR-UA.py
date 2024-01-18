import torch
from torch import nn

from src.models.modules.TCN import TCN
from src.models.modules.positional_embedding import PositionalEmbedding
from src.eval.soft_auroc import AUROC, soft_AUROC
from src.utils import min_max_normalization


class NAVAR_UA(nn.Module):
    """
    Implements the Neural Additive Vector Auto-Regression (NAVAR) model with uncertainty.

    Args:
        n_variables (int): Number of variables in the time series.
        hidden_dim (int): Hidden dimension size for the model.
        lambda1 (float, optional): Regularization factor.
        n_ensembles (int): Number of ensembles per dataset. Default is 1 for not using ensembles.
        **kwargs: Additional keyword arguments for the underlying :class:`src.models.tcn.TCN` architecture.

    Methods:
        forward(x): Forward pass through the NAVAR model.
        loss_function(y_true, prediction, contributions): Compute the loss for training.
        analysis(prediction, contributions): Perform analysis on the results.
    """

    def __init__(self, n_variables, hidden_dim, lambda1, beta, n_ensembles=1, **kwargs):
        super().__init__()
        self.lambda1 = lambda1
        self.beta = beta
        self.n_variables = n_variables
        self.n_ensembles = n_ensembles
        self.groups = n_variables * n_ensembles

        self.contributions = nn.Sequential(
            # Up-sample and add positional embedding
            PositionalEmbedding(n_variables, hidden_dim, n_ensembles),
            # Initialize the TCN model to learn additive contributions.
            TCN(
                in_channels=self.groups * hidden_dim,
                out_channels=self.groups * n_variables * 2,
                hidden_dim=self.groups * hidden_dim,
                groups=self.groups,
                **kwargs
            )
        )

        # Initialize learnable biases with small values.
        self.biases = nn.Parameter(torch.randn(1, n_ensembles, n_variables, 1) * 0.01)

    def forward(self, x, x_noise_adjusted=None, return_causal_matrix=False, ground_truth=None):
        """
        Forward pass through the NAVAR model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_variables, sequence_length).
            x_noise_adjusted (torch.Tensor, optional): Tensor of true mean values of the time series.
                Shape: (batch_size, n_variables, sequence_length).
            return_causal_matrix (bool): Flag indicating whether to return the causal matrix.
            ground_truth (torch.Tensor, optional): Ground truth tensor for the causal matrix.

        Returns:
            dict: A dictionary containing various outputs like predictions, loss, and optional metrics.
        """
        batch_size, n_var, seq_len = x.size()

        # Calculate contributions and reshape it.
        mean_var = self.contributions(x).reshape(batch_size, n_var, self.n_ensembles, n_var * 2, -1)

        mean, var = mean_var.chunk(2, dim=-2)
        var = nn.functional.softplus(var)
        contributions = mean
        if self.training:
            contributions = mean + torch.randn_like(var) * var

        # Sum contributions along dimension 1 and add biases to get the final prediction.
        prediction = contributions.sum(dim=1) + self.biases

        return self.process(x, prediction, contributions, var, x_noise_adjusted,
                            return_causal_matrix, ground_truth)

    def process(self, x, prediction, contributions, contribution_var, x_noise_adjusted=None,
                return_causal_matrix=False, ground_truth=None):
        """
        Processes the outputs of the forward pass, computing losses and other metrics.

        Args:
            x (torch.Tensor): Original input tensor of size (batch_size, n_var, seq_len).
            prediction (torch.Tensor): Prediction tensor from the model
                of size (batch_size, n_ensembles, n_var, seq_len).
            contributions (torch.Tensor): Contributions tensor from the model
                of size (batch_size, n_var, n_ensembles, n_var, seq_len).
            contribution_var (torch.Tensor): Var Contributions tensor from the model
                of size (batch_size, n_var, n_ensembles, n_var, seq_len).
            x_noise_adjusted (torch.Tensor, optional): Tensor of true mean values of the time series
                of size (batch_size, n_var, seq_len).
            return_causal_matrix (bool): Flag indicating whether to return the causal matrix.
            ground_truth (torch.Tensor, optional): Ground truth tensor for the causal matrix is of size (n_var, n_var).

        Returns:
            dict: A dictionary containing the loss, prediction, and optional metrics like causal matrix and AUROC.
        """
        s = contributions.size(-1)
        error = (x[:, None, :, 1 - s:] - prediction[..., :-1]).pow(2)
        prediction = prediction.detach().mean(dim=1)  # (batch_size, n_var, seq_len)
        # var: (batch_size, n_var, n_ensembles, n_var, -1)
        total_var = contribution_var[..., :-1].sum(dim=1)
        nll_loss = 0.5 * (total_var.log() + error / total_var)
        if self.beta > 0:
            nll_loss = torch.mean(nll_loss * total_var.detach().pow(self.beta))

        regularization_loss = self.lambda1 * contributions.abs().mean()

        # to size (n_ensembles, n_var, n_var, seq_len)
        contributions = contributions.detach().squeeze(0).permute(1, 2, 0, 3)
        contribution_var = contribution_var.detach().squeeze(0).permute(1, 2, 0, 3)

        result = {
            'loss': nll_loss + regularization_loss,
            'prediction': prediction.squeeze(0)
        }

        # Additional computations if noise-adjusted values are provided
        if x_noise_adjusted is not None:
            result['noise_adjusted_regression_loss'] = nn.functional.mse_loss(x_noise_adjusted[..., 1 - s:],
                                                                              prediction[..., :-1])

        # Compute causal matrix and AUROC if needed
        if return_causal_matrix or ground_truth is not None:
            assert not self.training, 'causal matrix should not be sampled from variational distribution, ' \
                                      'use model.eval()'
            contributions_mean = contributions.mean(dim=0)  # (n_var, n_var, seq_len)
            contributions_ep = contributions.std(dim=0)  # (n_var, n_var, seq_len)
            contributions_al = contribution_var.mean(dim=0)  # (n_var, n_var, seq_len)

            matrix_mean = contributions_mean.std(dim=-1)  # (n_var, n_var)
            matrix_ep = contributions_ep.mean(dim=-1)  # (n_var, n_var)
            matrix_al = contributions_al.mean(dim=-1)  # (n_var, n_var)

            matrix_mean = min_max_normalization(matrix_mean, min_val=0.0, max_val=1.0)

            result = {
                'contributions': contributions_mean,
                'contributions_ep': contributions_ep,
                'contributions_al': contributions_al,
                'matrix': matrix_mean,
                'matrix_ep': matrix_ep,
                'matrix_al': matrix_al,
                **result
            }

            if ground_truth is not None:
                ground_truth = ground_truth.to(matrix_mean.device)

                auc, tpr, fpr = AUROC(ground_truth, matrix_mean)
                result = {**result, 'AUROC': auc, 'TPR': tpr, 'FPR': fpr}

                auc, tpr, fpr = soft_AUROC(ground_truth, matrix_mean, matrix_ep)
                result = {**result, 'soft_AUROC': auc, 'soft_TPR': tpr, 'soft_FPR': fpr}

        return result


def main():
    # Parameters for the model
    n_variables = 3
    hidden_dim = 16
    lambda1 = 0.1
    beta = 0.5
    tcn_params = {'n_blocks': 2, 'n_layers': 2, 'kernel_size': 2, 'dropout': 0.2}

    # Initialize the NAVAR model
    model = NAVAR_UA(n_variables, hidden_dim, lambda1, beta, n_ensembles=17, **tcn_params).eval()

    # Generate dummy input data (batch_size, n_variables, sequence_length)
    batch_size = 1
    sequence_length = 20
    x = torch.randn(batch_size, n_variables, sequence_length)

    # Run the model
    output = model.forward(x, x_noise_adjusted=torch.randn(batch_size, n_variables, sequence_length), ground_truth=torch.randn(3, 3)>0)

    # Print the results
    for k in output.keys():
        print(f"{k}:", output[k].item() if output[k].numel() == 1 else output[k].shape)


if __name__ == "__main__":
    main()
