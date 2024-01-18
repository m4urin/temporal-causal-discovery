import torch
from torch import nn

from src.models.modules.TCN import TCN
from src.models.modules.positional_embedding import PositionalEmbedding
from src.eval.soft_auroc import AUROC
from src.utils import min_max_normalization


class TAMCaD(nn.Module):
    def __init__(self, n_variables, hidden_dim, lambda1, beta, gamma, **kwargs):
        super().__init__()
        self.gamma = gamma  # continuous matrices
        self.hidden_dim = hidden_dim

        self.tcn = nn.Sequential(
            # Up-sample and add positional embedding
            PositionalEmbedding(n_variables, hidden_dim),
            # Initialize the TCN model to learn additive contributions.
            TCN(
                in_channels=n_variables * hidden_dim,
                out_channels=n_variables * (hidden_dim + n_variables),
                hidden_dim=n_variables * hidden_dim,
                groups=n_variables,
                **kwargs
            )
        )
        self.prediction = nn.Sequential(
            nn.Conv1d(in_channels=n_variables * hidden_dim,
                      out_channels=n_variables * (hidden_dim // 2),
                      kernel_size=1, groups=n_variables),
            nn.ReLU(),
            nn.Conv1d(in_channels=n_variables * (hidden_dim // 2),
                      out_channels=n_variables,
                      kernel_size=1, groups=n_variables)
        )

    def forward(self, x, x_noise_adjusted=None, return_causal_matrix=False, temporal_matrix=False,
                ground_truth=None, mask=None):

        batch_size, n_var, seq_len = x.size()

        context, attention_logits = self.tcn(x)\
            .reshape(batch_size, n_var, n_var + self.hidden_dim, -1)\
            .split([self.hidden_dim, n_var], dim=2)

        # Apply masking if provided
        if mask is None:
            attentions = torch.softmax(attention_logits, dim=2)
        else:
            attentions = torch.softmax(torch.masked_fill(attention_logits, mask, -1e9), dim=2)

        # x: (batch_size, n_var * hidden_dim, sequence_length)
        z = torch.einsum('bijt, bjdt -> bidt', attentions, context).reshape(batch_size, n_var * self.hidden_dim, -1)

        prediction = self.prediction(z)

        return self.process(x, prediction, attention_logits, x_noise_adjusted,
                            return_causal_matrix, temporal_matrix, ground_truth)

    def process(self, x, prediction, attention_logits, x_noise_adjusted=None,
                return_causal_matrix=False, temporal_matrix=False, ground_truth=None):
        """
        Processes the outputs of the forward pass, computing losses and other metrics.

        Args:
            x (torch.Tensor): Original input tensor of size (batch_size, n_var, seq_len).
            prediction (torch.Tensor): Prediction tensor from the model
                of size (batch_size, n_var, seq_len).
            attention_logits (torch.Tensor): attention_logits tensor from the model
                of size (batch_size, n_var, n_var, seq_len).
            x_noise_adjusted (torch.Tensor, optional): Tensor of true mean values of the time series
                of size (batch_size, n_var, seq_len).
            return_causal_matrix (bool): Flag indicating whether to return the causal matrix.
            temporal_matrix (bool): Flag to use sliding window for temporal matrices.
            ground_truth (torch.Tensor, optional): Ground truth tensor for the causal matrix an is
                of size (n_var, n_var) or (n_var, n_var, seq_len), corresponding with temporal_matrix flag.

        Returns:
            dict: A dictionary containing the loss, prediction, and optional metrics like causal matrix and AUROC.
        """
        s = attention_logits.size(-1)
        regression_loss = nn.functional.mse_loss(x[..., 1 - s:], prediction[..., :-1])
        prediction = prediction.detach()  # (1, n_var, seq_len)

        regularization_continuous = self.gamma * torch.diff(attention_logits, dim=-1).abs().mean()

        attention_logits = attention_logits.detach().squeeze(0)  # (n_var, n_var, seq_len)

        result = {
            'loss': regression_loss + regularization_continuous,
            'prediction': prediction.squeeze(0),
            'attention_logits': attention_logits
        }

        # Additional computations if noise-adjusted values are provided
        if x_noise_adjusted is not None:
            result['noise_adjusted_regression_loss'] = nn.functional.mse_loss(x_noise_adjusted[..., 1 - s:],
                                                                              prediction[..., :-1])

        # Compute causal matrix and AUROC if needed
        if return_causal_matrix or ground_truth is not None:
            causal_matrix = min_max_normalization(attention_logits, min_val=0.0, max_val=1.0)
            if not temporal_matrix:
                causal_matrix = causal_matrix.mean(dim=-1)
            result['matrix'] = causal_matrix

            if ground_truth is not None:
                if temporal_matrix:
                    ground_truth = ground_truth[..., 1 - s:]
                    causal_matrix = causal_matrix[..., :-1]
                ground_truth = ground_truth.to(causal_matrix.device)

                auc, tpr, fpr = AUROC(ground_truth, causal_matrix)
                result = {**result, 'AUROC': auc, 'TPR': tpr, 'FPR': fpr}

        return result


def main():
    # Parameters for the model
    n_variables = 3
    hidden_dim = 16
    lambda1, beta, gamma = 0.1, 0.1, 0.1
    tcn_params = {'n_blocks': 2, 'n_layers': 2, 'kernel_size': 2, 'dropout': 0.2}

    # Initialize the NAVAR model
    model = TAMCaD(n_variables, hidden_dim, lambda1, beta, gamma, **tcn_params)

    # Generate dummy input data (batch_size, n_variables, sequence_length)
    batch_size = 1
    sequence_length = 20
    x = torch.randn(batch_size, n_variables, sequence_length)

    # Run the model
    output = model.forward(x,
                           return_causal_matrix=True,
                           ground_truth=torch.randn(n_variables, n_variables, sequence_length) > 0,
                           temporal_matrix=True)

    # Print the results
    for k in output.keys():
        print(f"{k}:", output[k].item() if output[k].numel() == 1 else output[k].shape)


if __name__ == "__main__":
    main()
