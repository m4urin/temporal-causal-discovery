import torch
from torch import nn

from src.models.modules.TCN import TCN
from src.models.modules.positional_embedding import PositionalEmbedding
from src.eval.soft_auroc import AUROC, soft_AUROC
from src.utils import min_max_normalization


class TAMCaD_UA(nn.Module):
    def __init__(self, n_variables, hidden_dim, lambda1, beta, gamma, p=0.7, n_ensembles=1, **kwargs):
        super().__init__()
        self.gamma = gamma  # continuous matrices
        self.hidden_dim = hidden_dim
        self.n_variables = n_variables
        self.n_ensembles = n_ensembles
        self.groups = n_variables * n_ensembles

        if n_ensembles > 1:
            self.bernoulli_mask = lambda1 * torch.bernoulli(torch.full((1, n_variables, n_ensembles, n_variables, 1), p))

        self.tcn = nn.Sequential(
            # Up-sample and add positional embedding
            PositionalEmbedding(n_variables, hidden_dim, n_ensembles),
            # Initialize the TCN model to learn additive contributions.
            TCN(
                in_channels=self.groups * hidden_dim,
                out_channels=self.groups * (hidden_dim + 2 * n_variables),
                hidden_dim=self.groups * hidden_dim,
                groups=self.groups,
                **kwargs
            )
        )
        self.prediction = nn.Sequential(
            nn.Conv1d(in_channels=self.groups * hidden_dim,
                      out_channels=self.groups * (hidden_dim // 2),
                      kernel_size=1, groups=self.groups),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.groups * (hidden_dim // 2),
                      out_channels=self.groups,
                      kernel_size=1, groups=self.groups)
        )

    def forward(self, x, x_noise_adjusted=None, return_causal_matrix=False, ground_truth=None, mask=None):

        batch_size, n_var, seq_len = x.size()

        context, attn_mean, attn_var = self.tcn(x)\
            .reshape(batch_size, n_var, self.n_ensembles, 2 * n_var + self.hidden_dim, -1)\
            .split([self.hidden_dim, n_var, n_var], dim=-2)

        s = context.size(-1)

        attn_var = nn.functional.softplus(attn_var)
        attention_logits = attn_mean
        if self.training:
            attention_logits = attn_mean + torch.randn_like(attn_var) * attn_var

        # Apply masking if provided
        if mask is None:
            attentions = torch.softmax(attention_logits, dim=-2)
        else:
            attentions = torch.softmax(torch.masked_fill(attention_logits, mask, -1e9), dim=-2)

        # x: (batch_size, n_var * hidden_dim, sequence_length)
        z = torch.einsum('biejt, bjedt -> biedt', attentions, context)\
            .reshape(batch_size, -1, s)

        prediction = self.prediction(z)\
            .reshape(1, n_var, self.n_ensembles, s).transpose(1, 2)  # (1, n_ensembles, n_var, seq_len)

        return self.process(x, prediction, attentions, attention_logits, attn_var, x_noise_adjusted,
                            return_causal_matrix, ground_truth)

    def process(self, x, prediction, attentions, attention_logits, attention_logits_var, x_noise_adjusted=None,
                return_causal_matrix=False, ground_truth=None):
        """
        Processes the outputs of the forward pass, computing losses and other metrics.

        Args:
            x (torch.Tensor): Original input tensor of size (batch_size, n_var, seq_len).
            prediction (torch.Tensor): Prediction tensor from the model
                of size (batch_size, n_ensembles, n_var, seq_len).
            attentions (torch.Tensor): attentions tensor from the model
                of size (batch_size, n_var, n_ensembles, n_var, seq_len).
            attention_logits (torch.Tensor): attention_logits tensor from the model
                of size (batch_size, n_var, n_ensembles, n_var, seq_len).
            attention_logits_var (torch.Tensor): Var Contributions tensor from the model
                of size (batch_size, n_var, n_ensembles, n_var, seq_len).
            x_noise_adjusted (torch.Tensor, optional): Tensor of true mean values of the time series
                of size (batch_size, n_var, seq_len).
            return_causal_matrix (bool): Flag indicating whether to return the causal matrix.
            ground_truth (torch.Tensor, optional): Ground truth tensor for the causal matrix is of size (n_var, n_var).

        Returns:
            dict: A dictionary containing the loss, prediction, and optional metrics like causal matrix and AUROC.
        """
        s = attention_logits.size(-1)
        regression_loss = (x[:, None, :, 1 - s:] - prediction[..., :-1]).pow(2).mean()
        prediction = prediction.detach().mean(dim=1)  # (1, n_var, seq_len)

        regularization = self.gamma * torch.diff(attention_logits, dim=-1).abs().mean()
        if self.n_ensembles > 1:
            regularization = regularization + torch.mean(self.bernoulli_mask * attentions)

        # to size (n_ensembles, n_var, n_var, seq_len)
        attention_logits = attention_logits.detach().squeeze(0).transpose(0, 1)  # (n_ens, n_var, n_var, seq_len)
        attention_logits_var = attention_logits_var.detach().squeeze(0).transpose(0, 1)  # (n_ens, n_var, n_var, seq_len)

        result = {
            'loss': regression_loss + regularization,
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

            attn_logits_mean = attention_logits.mean(dim=0)  # (n_var, n_var, seq_len)
            attn_logits_ep = attention_logits.std(dim=0)  # (n_var, n_var, seq_len)
            attn_logits_al = attention_logits_var.mean(dim=0)  # (n_var, n_var, seq_len)

            matrix_mean = attn_logits_mean.mean(dim=-1)  # (n_var, n_var)
            matrix_ep = attn_logits_ep.mean(dim=-1)  # (n_var, n_var)
            matrix_al = attn_logits_al.mean(dim=-1)  # (n_var, n_var)

            matrix_mean = min_max_normalization(matrix_mean, min_val=0.0, max_val=1.0)

            result = {
                'attention_logits': attn_logits_mean,
                'attention_logits_ep': attn_logits_ep,
                'attention_logits_al': attn_logits_al,
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
    lambda1, beta, gamma = 0.1, 0.1, 0.1
    tcn_params = {'n_blocks': 2, 'n_layers': 2, 'kernel_size': 2, 'dropout': 0.2}

    # Initialize the NAVAR model
    model = TAMCaD_UA(n_variables, hidden_dim, lambda1, beta, gamma, n_ensembles=17, **tcn_params).eval()

    # Generate dummy input data (batch_size, n_variables, sequence_length)
    batch_size = 1
    sequence_length = 20
    x = torch.randn(batch_size, n_variables, sequence_length)

    # Run the model
    output = model.forward(x,
                           return_causal_matrix=True,
                           ground_truth=torch.randn(n_variables, n_variables) > 0,
                           x_noise_adjusted=torch.randn(batch_size, n_variables, sequence_length))

    # Print the results
    for k in output.keys():
        print(f"{k}:", output[k].item() if output[k].numel() == 1 else output[k].shape)


if __name__ == "__main__":
    main()
