import torch
from torch import nn

from src.models.TCN import TCN
from src.eval.sliding_window_std import sliding_window_std
from src.eval.soft_auroc import AUROC
from src.utils import min_max_normalization


class NAVAR(nn.Module):
    """
    Implements the Neural Additive Vector Auto-Regression (NAVAR) model.

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

    def __init__(self, n_variables, hidden_dim, lambda1, **kwargs):
        super().__init__()
        self.lambda1 = lambda1

        self.contributions = TCN(
            in_channels=n_variables,
            out_channels=n_variables * n_variables,
            hidden_dim=n_variables * hidden_dim,
            groups=n_variables,
            **kwargs
        )

        # Initialize learnable biases with small values.
        self.biases = nn.Parameter(torch.randn(1, n_variables, 1) * 0.01)

    def forward(self, x, x_noise_adjusted=None, create_artifacts=False, temporal_matrix=False, ground_truth=None):
        """
        Forward pass through the NAVAR model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_variables, sequence_length).
            x_noise_adjusted (torch.Tensor, optional): Tensor of true mean values of the time series.
                Shape: (batch_size, n_variables, sequence_length).
            create_artifacts (bool): Flag indicating whether to return the causal matrix.
            temporal_matrix (bool): Flag to use sliding window for temporal matrices.
            ground_truth (torch.Tensor, optional): Ground truth tensor for the causal matrix.

        Returns:
            dict: A dictionary containing various outputs like predictions, loss, and optional metrics.
        """
        batch_size, n_var, seq_len = x.size()

        # Calculate contributions and reshape it.
        contributions = self.contributions(x).reshape(batch_size, n_var, n_var, -1)

        # Sum contributions along dimension 1 and add biases to get the final prediction.
        prediction = contributions.sum(dim=1) + self.biases

        return self.process(x, prediction, contributions, x_noise_adjusted,
                            create_artifacts, temporal_matrix, ground_truth)

    def process(self, x, prediction, contributions, x_noise_adjusted, create_artifacts, temporal_matrix, ground_truth):
        """
        Processes the outputs of the forward pass, computing losses and other metrics.

        Args:
            x (torch.Tensor): Original input tensor of size (batch_size, n_var, seq_len).
            prediction (torch.Tensor): Prediction tensor from the model
                of size (batch_size, n_var, seq_len).
            contributions (torch.Tensor): Contributions tensor from the model
                of size (batch_size, n_var, n_var, seq_len).
            x_noise_adjusted (torch.Tensor, optional): Tensor of true mean values of the time series
                of size (batch_size, n_var, seq_len).
            create_artifacts (bool): Flag indicating whether to return the artifacts.
            temporal_matrix (bool): Flag to use sliding window for temporal matrices.
            ground_truth (torch.Tensor, optional): Ground truth tensor for the causal matrix an is
                of size (n_var, n_var) or (n_var, n_var, seq_len), corresponding with temporal_matrix flag.

        Returns:
            dict: A dictionary containing the loss, prediction, and optional metrics like causal matrix and AUROC.
        """
        metrics, artifacts = {}, {}
        s = contributions.size(-1)

        regression_loss = nn.functional.mse_loss(x[..., 1 - s:], prediction[..., :-1])
        prediction = prediction.detach()  # (1, n_var, seq_len)

        regularization_loss = self.lambda1 * contributions.abs().mean()
        contributions = contributions.detach().transpose(1, 2)  # (bs, n_var, n_var, seq_len)
        metrics['loss'] = regression_loss + regularization_loss

        if create_artifacts:
            artifacts = {
                **artifacts,
                'prediction': prediction,
                'contributions': contributions
            }

        # Additional computations if noise-adjusted values are provided
        if x_noise_adjusted is not None:
            metrics['noise_adjusted_regression_loss'] = nn.functional.mse_loss(x_noise_adjusted[..., 1 - s:],
                                                                               prediction[..., :-1])

        # Compute causal matrix and AUROC if needed
        if create_artifacts or ground_truth is not None:
            if temporal_matrix:
                causal_matrix = sliding_window_std(contributions, window=(30, 30), dim=-1)
            else:
                causal_matrix = contributions.std(dim=-1)
            causal_matrix = causal_matrix.mean(dim=0)

            causal_matrix = min_max_normalization(causal_matrix, min_val=0.0, max_val=1.0)
            if create_artifacts:
                artifacts['matrix'] = causal_matrix

            if ground_truth is not None:
                if temporal_matrix:
                    ground_truth = ground_truth[..., 1 - s:]
                    causal_matrix = causal_matrix[..., :-1]
                ground_truth = ground_truth.to(causal_matrix.device)
                auc, tpr, fpr = AUROC(ground_truth, causal_matrix)
                metrics.update({'AUROC': auc})
                if create_artifacts:
                    artifacts.update({'TPR': tpr, 'FPR': fpr})

        return metrics, artifacts


def main():
    # Parameters for the model
    n_variables = 3
    hidden_dim = 16
    lambda1 = 0.1
    tcn_params = {'n_blocks': 2, 'n_layers': 2, 'kernel_size': 2, 'dropout': 0.2}

    # Initialize the NAVAR model
    model = NAVAR(n_variables, hidden_dim, lambda1, **tcn_params)

    # Generate dummy input data (batch_size, n_variables, sequence_length)
    batch_size = 1
    sequence_length = 2000
    x = torch.randn(batch_size, n_variables, sequence_length)

    # Run the model
    metrics, artifacts = model.forward(x,
                                       create_artifacts=True)

    # Print the results
    print('Metrics:')
    for k in metrics.keys():
        print(f"{k}:", metrics[k].item() if metrics[k].numel() == 1 else metrics[k].shape)
    if len(artifacts) > 0:
        print('\nArtifacts:')
        for k in artifacts.keys():
            print(f"{k}:", artifacts[k].item() if artifacts[k].numel() == 1 else artifacts[k].shape)

    torch.save({**metrics, **artifacts}, 'test.pt')


if __name__ == "__main__":
    main()
