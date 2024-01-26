import torch
from torch import nn

from src.eval.sliding_window_std import sliding_window_std
from src.models.TCN import TCN
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

        self.contributions = TCN(
                in_channels=self.groups,
                out_channels=self.groups * n_variables * 2,
                hidden_dim=self.groups * hidden_dim,
                groups=self.groups,
                **kwargs
            )

        # Initialize learnable biases with small values.
        self.biases = nn.Parameter(torch.randn(1, n_ensembles, n_variables, 1) * 0.01)

    def forward(self, x, x_noise_adjusted=None, create_artifacts=False, ground_truth=None, temporal_matrix=False):
        """
        Forward pass through the NAVAR model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_variables, sequence_length).
            x_noise_adjusted (torch.Tensor, optional): Tensor of true mean values of the time series.
                Shape: (batch_size, n_variables, sequence_length).
            create_artifacts (bool): Flag indicating whether to return artifacts.
            ground_truth (torch.Tensor, optional): Ground truth tensor for the causal matrix.

        Returns:
            dict: A dictionary containing various outputs like predictions, loss, and optional metrics.
        """
        batch_size, n_var, seq_len = x.size()

        # Calculate contributions and reshape it.
        mean_var = self.contributions(x.repeat(1, self.n_ensembles, 1))\
            .reshape(batch_size, self.n_ensembles, n_var, n_var * 2, -1)

        contribution_mean, contribution_var = mean_var.chunk(2, dim=-2)
        contribution_var = nn.functional.softplus(contribution_var)
        contributions = contribution_mean
        if self.training:
            contributions = contribution_mean + torch.randn_like(contribution_var) * contribution_var

        # Sum contributions along dimension 1 and add biases to get the final prediction.
        prediction = contributions.sum(dim=2) + self.biases

        return self.process(x, prediction, contribution_mean, contribution_var, x_noise_adjusted, create_artifacts, ground_truth, temporal_matrix)

    def process(self, x, prediction, contributions, contribution_var, x_noise_adjusted, create_artifacts, ground_truth, temporal_matrix):
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
            create_artifacts (bool): Flag indicating whether to return the artifacts.
            ground_truth (torch.Tensor, optional): Ground truth tensor for the causal matrix is of size (n_var, n_var).

        Returns:
            dict: A dictionary containing the loss, prediction, and optional metrics like causal matrix and AUROC.
        """
        metrics, artifacts = {}, {}

        s = contributions.size(-1)
        # error: (batch_size, n_ensembles, n_var, seq_len)
        error = (x[:, None, :, 1 - s:] - prediction[..., :-1]).pow(2)
        prediction = prediction.detach().mean(dim=1)  # (batch_size, n_var, seq_len)
        # var: (batch_size, n_ensembles, n_var, n_var, -1)
        # var: (batch_size, n_ensembles, n_var, -1)
        total_var = contribution_var[..., :-1].sum(dim=2)
        nll_loss = 0.5 * (total_var.log() + error / total_var)
        if self.beta > 0:
            nll_loss *= total_var.detach().pow(self.beta)
        nll_loss = nll_loss.mean()

        regularization_loss = self.lambda1 * contributions.abs().mean()

        # contributions: (batch_size, n_ensembles, n_var, n_var, seq_len)
        # to size (n_ensembles, n_var, n_var, seq_len)
        contributions = contributions.detach().squeeze(0).permute(0, 2, 1, 3)
        contribution_var = contribution_var.detach().squeeze(0).permute(0, 2, 1, 3)

        metrics['loss'] = nll_loss + regularization_loss
        if create_artifacts:
            artifacts = {
                'prediction': prediction.squeeze(0)
            }

        # Additional computations if noise-adjusted values are provided
        if x_noise_adjusted is not None:
            metrics['noise_adjusted_regression_loss'] = nn.functional.mse_loss(x_noise_adjusted[..., 1 - s:],
                                                                               prediction[..., :-1])

        # Compute causal matrix and AUROC if needed
        if create_artifacts or ground_truth is not None:

            contributions_mean = contributions.mean(dim=0)  # (n_var, n_var, seq_len)
            contributions_ep = contributions.std(dim=0)  # (n_var, n_var, seq_len)
            contributions_al = contribution_var.mean(dim=0)  # (n_var, n_var, seq_len)

            matrix_mean = contributions_mean.std(dim=-1)  # (n_var, n_var)
            matrix_ep = contributions_ep.mean(dim=-1)  # (n_var, n_var)
            matrix_al = contributions_al.mean(dim=-1)  # (n_var, n_var)
            matrix_mean = min_max_normalization(matrix_mean, min_val=0.0, max_val=1.0)
            matrix_temporal = min_max_normalization(
                        sliding_window_std(contributions_mean, window=(30, 30), dim=-1),
                        min_val=0.0, max_val=1.0)
            contributions_ep = min_max_normalization(contributions_ep, min_val=0.0, max_val=1.0)
            if create_artifacts:
                artifacts.update({
                    'contributions': contributions_mean,
                    'contributions_ep': contributions_ep,
                    'contributions_al': contributions_al,
                    'matrix': matrix_mean,
                    'matrix_ep': matrix_ep,
                    'matrix_al': matrix_al,
                    'matrix_temporal': matrix_temporal
                })

            if ground_truth is not None:
                eval_matrix = matrix_mean
                eval_matrix_ep = matrix_ep
                if temporal_matrix:
                    eval_matrix = matrix_temporal[..., :-1]
                    eval_matrix_ep = contributions_ep[..., :-1]
                    ground_truth = ground_truth[..., 1 - s:]
                ground_truth = ground_truth.to(eval_matrix.device)

                auc, tpr, fpr = AUROC(ground_truth, eval_matrix)
                soft_auc, soft_tpr, soft_fpr = soft_AUROC(ground_truth, eval_matrix, eval_matrix_ep)

                metrics.update({'AUROC': auc, 'soft_AUROC': soft_auc})
                if create_artifacts:
                    artifacts.update({'TPR': tpr, 'FPR': fpr, 'soft_TPR': soft_tpr, 'soft_FPR': soft_fpr})

        return metrics, artifacts


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
    metrics, artifacts = model.forward(x,
                                       #x_noise_adjusted=torch.randn(batch_size, n_variables, sequence_length),
                                       #ground_truth=torch.randn(3, 3) > 0,
                                       create_artifacts=True
                                       )

    # Print the results
    print('Metrics:')
    for k in metrics.keys():
        print(f"{k}:", metrics[k].item() if metrics[k].numel() == 1 else metrics[k].shape)
    if len(artifacts) > 0:
        print('\nArtifacts:')
        for k in artifacts.keys():
            print(f"{k}:", artifacts[k].item() if artifacts[k].numel() == 1 else artifacts[k].shape)


if __name__ == "__main__":
    main()
