import torch
from torch import nn

from src.models.TCN import TCN
from src.eval.soft_auroc import AUROC
from src.utils import min_max_normalization


class TAMCaD_T(nn.Module):
    def __init__(self, n_variables, hidden_dim, gamma, **kwargs):
        super().__init__()
        self.gamma = gamma  # continuous matrices
        self.hidden_dim = hidden_dim

        self.qkv = TCN(
                in_channels=n_variables,
                out_channels=n_variables * (3 * hidden_dim),  # q, k, v
                hidden_dim=n_variables * hidden_dim,
                groups=n_variables,
                **kwargs
        )

        self.dot_product_attention = ScaledDotProductAttention(n_heads=1)

        self.prediction = nn.Sequential(
            nn.Conv1d(in_channels=n_variables * hidden_dim,
                      out_channels=n_variables * (hidden_dim // 2),
                      kernel_size=1, groups=n_variables),
            nn.ReLU(),
            nn.Conv1d(in_channels=n_variables * (hidden_dim // 2),
                      out_channels=n_variables,
                      kernel_size=1, groups=n_variables)
        )

    def forward(self, x, x_noise_adjusted=None, create_artifacts=False, temporal_matrix=False, ground_truth=None,
                mask=None):

        batch_size, n_var, seq_len = x.size()

        # q, k, v: (batch_size, n_var, hidden_dim, seq_len)
        q, k, v = self.qkv(x).reshape(batch_size, n_var, 3 * self.hidden_dim, -1).chunk(3, dim=2)
        # z: (batch_size, n_var * hidden_dim, sequence_length)
        # attn: (batch_size, n_var, n_var, seq_len)
        z, attentions, attention_logits = self.dot_product_attention(q, k, v, mask=mask)

        # prediction: (batch_size, n_var, sequence_length)
        prediction = self.prediction(z)

        return self.process(x, prediction, attention_logits, q, k, v, x_noise_adjusted,
                            create_artifacts, temporal_matrix, ground_truth)

    def process(self, x, prediction, attention_logits, q, k, v, x_noise_adjusted,
                create_artifacts, temporal_matrix, ground_truth):
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
            create_artifacts (bool): Flag indicating whether to return the artifacts.
            temporal_matrix (bool): Flag to use sliding window for temporal matrices.
            ground_truth (torch.Tensor, optional): Ground truth tensor for the causal matrix an is
                of size (n_var, n_var) or (n_var, n_var, seq_len), corresponding with temporal_matrix flag.

        Returns:
            dict: A dictionary containing the loss, prediction, and optional metrics like causal matrix and AUROC.
        """
        metrics, artifacts = {}, {}

        s = attention_logits.size(-1)
        loss = nn.functional.mse_loss(x[..., 1 - s:], prediction[..., :-1])
        prediction = prediction.detach()  # (1, n_var, seq_len)

        attention_logits = attention_logits.detach()  # (n_var, n_var, seq_len)

        if self.gamma > 0:
            loss = loss + self.gamma * torch.diff(attention_logits, dim=-1).abs().mean()

        metrics['loss'] = loss
        if create_artifacts:
            artifacts = {
                'prediction': prediction.squeeze(0),
                'attention_logits': attention_logits
            }
        if create_artifacts:
            artifacts.update({
                'q': q.detach()[0].mean(dim=-1),  # (n_var, hidden_dim)
                'k': k.detach()[0].mean(dim=-1),  # (n_var, hidden_dim)
                'v': v.detach()[0].mean(dim=-1)  # (n_var, hidden_dim)
            })

        # Additional computations if noise-adjusted values are provided
        if x_noise_adjusted is not None:
            metrics['noise_adjusted_regression_loss'] = nn.functional.mse_loss(x_noise_adjusted[..., 1 - s:],
                                                                               prediction[..., :-1])

        # Compute causal matrix and AUROC if needed
        if create_artifacts or ground_truth is not None:
            causal_matrix = torch.softmax(attention_logits * 1.2, dim=1)
            if not temporal_matrix:
                causal_matrix = causal_matrix.mean(dim=-1)
            causal_matrix = causal_matrix.mean(dim=0)

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


class ScaledDotProductAttention(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, q, k, v, mask=None):
        """
        Perform scaled dot-product attention for temporal (instantaneous0 attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, groups, hidden_dim, sequence_length).
            k (torch.Tensor): Key tensor of shape (batch_size, groups, hidden_dim, sequence_length).
            v (torch.Tensor): Value tensor of shape (batch_size, groups, hidden_dim, sequence_length).
            mask (torch.Tensor): Mask

        Returns:
            torch.Tensor: Output tensor after attention calculation of size (batch_size, n_var * hidden_dim, sequence_length).
            torch.Tensor: Attention weights of size (batch_size, n_var, n_var, sequence_length).
        """

        batch_size, groups_q, hidden_dim, sequence_length = q.size()
        groups_kv = k.size(1)
        d_k = hidden_dim // self.n_heads
        scale = d_k ** -0.5

        # Reshape q, k, and v tensors
        # (batch_size, n_heads, seq_length, groups_q, dk)
        q = q.reshape(batch_size, groups_q, self.n_heads, d_k, sequence_length).permute(0, 2, 4, 1, 3)
        # (batch_size, n_heads, seq_length, dk, groups_kv)
        k = k.reshape(batch_size, groups_kv, self.n_heads, d_k, sequence_length).permute(0, 2, 4, 3, 1)
        # (batch_size, n_heads, seq_length, groups_kv, dk)
        v = v.reshape(batch_size, groups_kv, self.n_heads, d_k, sequence_length).permute(0, 2, 4, 1, 3)

        # Calculate attention scores: (batch_size, num_heads, sequence_length, groups_q, groups_kv)
        attention_logits = scale * torch.matmul(q, k)

        attentions = attention_logits
        # Apply masking if provided
        if mask is not None:
            attentions = attention_logits.masked_fill(mask == 0, -1e9)

        # Calculate attention weights: (batch_size, num_heads, sequence_length, groups_q, groups_kv)
        attentions = torch.softmax(attentions, dim=-1)

        # Calculate output projection: (batch_size, num_heads, sequence_length, groups_q, dk)
        x = torch.matmul(attentions, v)

        # Rearrange: (batch_size, num_heads, groups_q, groups_kv, sequence_length)
        attentions = attentions.permute(0, 1, 3, 4, 2)
        attention_logits = attention_logits.permute(0, 1, 3, 4, 2)
        # Mean over heads: (batch_size, groups_q, groups_kv, sequence_length)
        attentions = attentions.mean(dim=1)
        attention_logits = attention_logits.mean(dim=1)

        # Permute to original form and concatenate heads: (batch_size, groups_q * hidden_dim, sequence_length)
        x = x.permute(0, 3, 1, 4, 2).reshape(batch_size, -1, sequence_length)

        return x, attentions, attention_logits


def main():
    # Parameters for the model
    n_variables = 3
    hidden_dim = 16
    lambda1, beta, gamma = 0.1, 0.1, 0.1
    tcn_params = {'n_blocks': 2, 'n_layers': 2, 'kernel_size': 2, 'dropout': 0.2}

    # Initialize the NAVAR model
    model = TAMCaD_T(n_variables, hidden_dim, gamma, **tcn_params)

    # Generate dummy input data (batch_size, n_variables, sequence_length)
    batch_size = 1
    sequence_length = 20
    x = torch.randn(batch_size, n_variables, sequence_length)

    # Run the model
    metrics, artifacts = model.forward(x, x_noise_adjusted=torch.randn_like(x),
                                       create_artifacts=True,
                                       #ground_truth=torch.randn(n_variables, n_variables, sequence_length) > 0,
                                       temporal_matrix=True)

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
