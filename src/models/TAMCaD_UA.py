import torch
from torch import nn

from src.models.TCN import TCN
from src.eval.soft_auroc import AUROC, soft_AUROC
from src.models.gumbel_softmax import GumbelSoftmax, SoftmaxModule
from src.utils import min_max_normalization


class TAMCaD_UA(nn.Module):
    def __init__(self, n_variables, hidden_dim, lambda1, gamma, n_ensembles, use_gumbel, p=0.5, **kwargs):
        super().__init__()
        self.gamma = gamma  # continuous matrices
        self.lambda1 = lambda1
        self.hidden_dim = hidden_dim
        self.n_variables = n_variables
        self.n_ensembles = n_ensembles
        self.groups = n_variables * n_ensembles

        if n_ensembles > 1:
            self.register_buffer('bernoulli_mask', torch.rand(1, n_ensembles, n_variables, n_variables, 1) < p)

        self.tcn = TCN(
                in_channels=self.groups,
                out_channels=self.groups * (hidden_dim + 2 * n_variables),
                hidden_dim=self.groups * hidden_dim,
                groups=self.groups,
                **kwargs
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
        self.softmax = GumbelSoftmax(temperature=0.9) if use_gumbel else SoftmaxModule()

    def forward(self, x, x_noise_adjusted=None, create_artifacts=False, ground_truth=None, mask=None, temporal_matrix=False):

        batch_size, n_var, seq_len = x.size()

        context, attn_mean, attn_var = self.tcn(x.repeat(1, self.n_ensembles, 1)) \
            .reshape(batch_size, self.n_ensembles, n_var, 2 * n_var + self.hidden_dim, -1) \
            .split([self.hidden_dim, n_var, n_var], dim=-2)

        s = context.size(-1)

        attn_var = nn.functional.softplus(attn_var) + 1e-6
        attention_logits = attn_mean
        if self.training:
            attention_logits = attn_mean + torch.randn_like(attn_var) * attn_var

        # Apply masking if provided
        if mask is None:
            attentions = self.softmax(attention_logits, dim=-2)
        else:
            attentions = self.softmax(torch.masked_fill(attention_logits, mask, -1e9), dim=-2)

        # x: (batch_size, n_var * hidden_dim, sequence_length)
        z = torch.einsum('beijt, bejdt -> beidt', attentions, context) \
            .reshape(batch_size, -1, s)

        prediction = self.prediction(z).reshape(-1, self.n_ensembles, n_var, s)  # (bs, n_ensembles, n_var, seq_len)

        return self.process(x, prediction, attentions, attn_mean, attn_var, x_noise_adjusted,
                            create_artifacts, ground_truth, temporal_matrix)

    def process(self, x, prediction, attentions, attention_logits, attention_logits_var,
                x_noise_adjusted, create_artifacts, ground_truth, temporal_matrix):
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
            create_artifacts (bool): Flag indicating whether to return the artifacts.
            ground_truth (torch.Tensor, optional): Ground truth tensor for the causal matrix is of size (n_var, n_var).
            temporal_matrix (bool): return causal matrix with time axis.

        Returns:
            dict: A dictionary containing the loss, prediction, and optional metrics like causal matrix and AUROC.
        """
        metrics, artifacts = {}, {}

        s = attention_logits.size(-1)
        regression_loss = (x[:, None, :, 1 - s:] - prediction[..., :-1]).pow(2).mean()
        prediction = prediction.detach().mean(dim=1)  # (1, n_var, seq_len)

        regularization = self.gamma * torch.diff(attention_logits, dim=-1).abs().mean()
        if self.n_ensembles > 1:
            mask_loss = torch.masked_fill(attentions, self.bernoulli_mask, value=0).mean()
            regularization = regularization + self.lambda1 * mask_loss

        # to size (n_ensembles, n_var, n_var, seq_len)
        attention_logits = attention_logits.detach().squeeze(0)  # (n_ens, n_var, n_var, seq_len)
        attention_logits_var = attention_logits_var.detach().squeeze(0)  # (n_ens, n_var, n_var, seq_len)

        metrics['loss'] = regression_loss + regularization
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

            attn_logits_mean = attention_logits.mean(dim=0)  # (n_var, n_var, seq_len)
            attn_logits_ep = attention_logits.std(dim=0)  # (n_var, n_var, seq_len)
            attn_logits_al = attention_logits_var.mean(dim=0)  # (n_var, n_var, seq_len)

            matrix_mean = attn_logits_mean.mean(dim=-1)  # (n_var, n_var)
            matrix_ep = attn_logits_ep.mean(dim=-1)  # (n_var, n_var)
            matrix_al = attn_logits_al.mean(dim=-1)  # (n_var, n_var)

            matrix_mean = min_max_normalization(matrix_mean, min_val=0.0, max_val=1.0)
            matrix_temporal = min_max_normalization(attn_logits_mean, min_val=0.0, max_val=1.0)
            if create_artifacts:
                artifacts.update({
                    'attention_logits': attn_logits_mean,
                    'attention_logits_ep': attn_logits_ep,
                    'attention_logits_al': attn_logits_al,
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
                    eval_matrix_ep = attn_logits_ep[..., :-1]
                    ground_truth = ground_truth[..., 1 - s:]
                ground_truth = ground_truth.to(eval_matrix.device)
                #ground_truth = ground_truth.unsqueeze(0).repeat(self.n_ensembles, 1, 1)
                auc, tpr, fpr = AUROC(ground_truth, eval_matrix)
                soft_auc, soft_tpr, soft_fpr = soft_AUROC(ground_truth, eval_matrix, eval_matrix_ep)

                metrics.update({'AUROC': auc, 'soft_AUROC': soft_auc})
                if create_artifacts:
                    artifacts.update({'TPR': tpr, 'FPR': fpr, 'soft_TPR': soft_tpr, 'soft_FPR': soft_fpr})

        return metrics, artifacts


def main():
    # Parameters for the model
    n_variables = 5
    hidden_dim = 32
    lambda1, beta, gamma = 0.1, 0.1, 0.1
    tcn_params = {'n_blocks': 2, 'n_layers': 2, 'kernel_size': 2, 'dropout': 0.2}

    # Initialize the NAVAR model
    model = TAMCaD_UA(n_variables, hidden_dim, lambda1, beta, gamma, n_ensembles=17, **tcn_params).eval()

    # Generate dummy input data (batch_size, n_variables, sequence_length)
    batch_size = 1
    sequence_length = 20
    x = torch.randn(batch_size, n_variables, sequence_length)

    # Run the model
    metrics, artifacts = model.forward(x,
                                       ground_truth=torch.randn(n_variables, n_variables) > 0,
                                       x_noise_adjusted=torch.randn(batch_size, n_variables, sequence_length))

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
