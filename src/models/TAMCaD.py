import torch
from torch import nn

from src.models.TCN import TCN
from src.eval.soft_auroc import AUROC
from src.models.gumbel_softmax import GumbelSoftmax, SoftmaxModule
from src.utils import min_max_normalization


class TAMCaD(nn.Module):
    def __init__(self, n_variables, hidden_dim, gamma, dropout, use_gumbel, **kwargs):
        super().__init__()
        self.gamma = gamma  # continuous matrices
        self.hidden_dim = hidden_dim
        self.softmax = GumbelSoftmax(temperature=0.9) if use_gumbel else SoftmaxModule()

        self.tcn = TCN(
                in_channels=n_variables,
                out_channels=n_variables * (hidden_dim + n_variables),
                hidden_dim=n_variables * hidden_dim,
                groups=n_variables,
                dropout=dropout,
                **kwargs
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

    def forward(self, x, x_noise_adjusted=None, create_artifacts=False,
                temporal_matrix=False, ground_truth=None, mask=None):

        batch_size, n_var, seq_len = x.size()

        context, attention_logits = self.tcn(x) \
            .reshape(batch_size, n_var, n_var + self.hidden_dim, -1) \
            .split([self.hidden_dim, n_var], dim=2)

        # Apply masking if provided
        if mask is None:
            attentions = self.softmax(attention_logits, dim=2)
        else:
            attentions = self.softmax(torch.masked_fill(attention_logits, mask, -1e9), dim=2)

        # x: (batch_size, n_var * hidden_dim, sequence_length)
        z = torch.einsum('bijt, bjdt -> bidt', attentions, context).reshape(batch_size, n_var * self.hidden_dim, -1)

        prediction = self.prediction(z)

        return self.process(x, prediction, attention_logits, x_noise_adjusted,
                            create_artifacts, temporal_matrix, ground_truth, attentions)

    def process(self, x, prediction, attention_logits, x_noise_adjusted,
                create_artifacts, temporal_matrix, ground_truth, attentions):
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
            create_artifacts (bool): Flag indicating whether to return artifacts.
            temporal_matrix (bool): Flag to use sliding window for temporal matrices.
            ground_truth (torch.Tensor, optional): Ground truth tensor for the causal matrix an is
                of size (n_var, n_var) or (n_var, n_var, seq_len), corresponding with temporal_matrix flag.
            attentions (torch.Tensor, optional): Attentions after applying softmax to the logits.
        Returns:
            dict: A dictionary containing the loss, prediction, and optional metrics like causal matrix and AUROC.
        """
        metrics, artifacts = {}, {}
        s = attention_logits.size(-1)
        loss = nn.functional.mse_loss(x[..., 1 - s:], prediction[..., :-1])
        prediction = prediction.detach()  # (1, n_var, seq_len)

        attention_logits = attention_logits.detach()  # (bs, n_var, n_var, seq_len)
        if self.gamma > 0:
            #loss = loss + self.gamma * torch.diff(attention_logits, dim=-1).abs().mean()
            #loss = loss + self.gamma * (-attentions * torch.log(attentions + 1e-10)).mean()
            loss = loss + self.gamma * (-attentions * torch.log(attentions + 1e-10)).mean()

        metrics['loss'] = loss
        if create_artifacts:
            artifacts = {
                'prediction': prediction.squeeze(0),
                'attention_logits': attention_logits
            }

        # Additional computations if noise-adjusted values are provided
        if x_noise_adjusted is not None:
            metrics['noise_adjusted_regression_loss'] = nn.functional.mse_loss(x_noise_adjusted[..., 1 - s:],
                                                                               prediction[..., :-1])

        # Compute causal matrix and AUROC if needed
        if create_artifacts or ground_truth is not None:
            causal_matrix = attention_logits
            if not temporal_matrix:
                causal_matrix = causal_matrix.mean(dim=-1)
            causal_matrix = causal_matrix.mean(dim=0)

            if create_artifacts:
                artifacts['matrix'] = min_max_normalization(causal_matrix, min_val=-1, max_val=1.0)

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
    lambda1, beta, gamma = 0.1, 0.1, 0.1
    tcn_params = {'n_blocks': 2, 'n_layers': 2, 'kernel_size': 2, 'dropout': 0.2}

    # Initialize the NAVAR model
    model = TAMCaD(n_variables, hidden_dim, gamma, **tcn_params)

    # Generate dummy input data (batch_size, n_variables, sequence_length)
    batch_size = 1
    sequence_length = 20
    x = torch.randn(batch_size, n_variables, sequence_length)

    # Run the model
    metrics, artifacts = model.forward(x,
                                       create_artifacts=True,
                                       ground_truth=torch.randn(n_variables, n_variables) > 0,
                                       temporal_matrix=False)

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
