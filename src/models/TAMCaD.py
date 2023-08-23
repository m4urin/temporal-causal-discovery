import torch
from torch import nn

from src.models.TCN import TCN
from src.models.categorical_distributions import get_softmax_method
from src.losses import TAMCaD_regularization_loss, DER_loss, NLL_loss
from src.utils import count_parameters, weighted_mean


class TAMCaD(nn.Module):
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block, n_heads,
                 softmax_method='softmax', dropout=0.0, weight_sharing=False, recurrent=False,
                 aleatoric=False, epistemic=False, **kwargs):
        super().__init__()
        if epistemic:
            self.tamcad = TAMCaD_Epistemic(n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block, n_heads,
                                           softmax_method, dropout, weight_sharing, recurrent)
        elif aleatoric:
            self.tamcad = TAMCaD_Aleatoric(n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block, n_heads,
                                           softmax_method, dropout, weight_sharing, recurrent)
        else:
            self.tamcad = TAMCaD_Default(n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block, n_heads,
                                         softmax_method, dropout, weight_sharing, recurrent)

        self.receptive_field = (2 ** n_blocks - 1) * n_layers_per_block * (kernel_size - 1) + 1
        self.n_params = count_parameters(self)

    def forward(self, x):
        return self.tamcad(x)

    def loss_function(self, y_true, **kwargs):
        return self.tamcad.loss_function(y_true, **kwargs)

    def analysis(self, **kwargs):
        mode = self.training
        self.eval()
        with torch.no_grad():
            result = self.tamcad.analysis(**kwargs)
            #result = {k: v.cpu().numpy() for k, v in result.items()}
        self.train(mode)
        return result


class TAMCaD_Default(nn.Module):
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                 n_heads=1, softmax_method='softmax', dropout=0.0, weight_sharing=False, recurrent=False):
        super().__init__()
        self.attention_mechanism = TemporalInstantaneousAttentionMechanism(
            in_channels=n_variables,
            out_channels=n_variables,
            hidden_dim=hidden_dim,
            groups=n_variables,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            n_heads=n_heads,
            softmax_method=softmax_method,
            weight_sharing=weight_sharing,
            recurrent=recurrent,
            dropout=dropout
        )

    def forward(self, x):
        prediction, attentions = self.attention_mechanism(x)
        return {
            'prediction': prediction,
            'attentions': attentions
        }

    @staticmethod
    def analysis(prediction, attentions):
        temp_causal_matrix, temp_external_causal_matrix = instant_attentions_to_causal_matrix(attentions)
        return {
            'prediction': prediction,
            'attentions': attentions,  # (batch_size, n_heads, n, 2*n, T)
            'default_causal_matrix': temp_causal_matrix.mean(dim=-1),
            'default_external_causal_matrix': temp_external_causal_matrix.mean(dim=-1),
            'temp_causal_matrix': temp_causal_matrix,
            'temp_external_causal_matrix': temp_external_causal_matrix,
        }

    @staticmethod
    def loss_function(y_true, prediction, attentions, lambda1=0.2, beta=0.2, coeff=None):
        # Mean squared error loss
        error = nn.functional.mse_loss(prediction, y_true)
        regularization = TAMCaD_regularization_loss(attentions, lambda1=lambda1, beta=beta)
        return error + regularization


class TAMCaD_Aleatoric(nn.Module):
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                 n_heads=1, softmax_method='softmax', dropout=0.0, weight_sharing=False, recurrent=False):
        super().__init__()
        self.attention_mechanism = TemporalInstantaneousAttentionMechanism(
            in_channels=n_variables,
            out_channels=2 * n_variables,
            hidden_dim=hidden_dim,
            groups=n_variables,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            n_heads=n_heads,
            softmax_method=softmax_method,
            weight_sharing=weight_sharing,
            recurrent=recurrent,
            dropout=dropout
        )

    def forward(self, x):
        batch_size, n, sequence_length = x.size()
        x, attentions = self.attention_mechanism(x)
        prediction, log_var_aleatoric = x.reshape(batch_size, n, 2, sequence_length).unbind(dim=2)
        return {
            'prediction': prediction,
            'attentions': attentions,
            'log_var_aleatoric': log_var_aleatoric
        }

    @staticmethod
    def analysis(prediction, attentions, log_var_aleatoric):
        temp_causal_matrix, temp_external_causal_matrix = instant_attentions_to_causal_matrix(attentions)
        return {
            'prediction': prediction,
            'attentions': attentions,  # (batch_size, n_heads, n, 2*n, T)
            'aleatoric': torch.exp(0.5 * log_var_aleatoric),
            'default_causal_matrix': temp_causal_matrix.mean(dim=-1),
            'default_external_causal_matrix': temp_external_causal_matrix.mean(dim=-1),
            'temp_causal_matrix': temp_causal_matrix,
            'temp_external_causal_matrix': temp_external_causal_matrix,
        }

    @staticmethod
    def loss_function(y_true, prediction, attentions, log_var_aleatoric, lambda1=0.2, beta=0.2, coeff=None):
        aleatoric_loss = NLL_loss(y_true=y_true, y_pred=prediction, log_var=log_var_aleatoric)
        regularization = TAMCaD_regularization_loss(attentions, lambda1=lambda1, beta=beta)
        return aleatoric_loss + regularization


class TAMCaD_Epistemic(nn.Module):
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                 n_heads=1, softmax_method='softmax', dropout=0.0, weight_sharing=False, recurrent=False):
        super().__init__()
        self.attention_mechanism = TemporalInstantaneousAttentionMechanism(
            in_channels=n_variables,
            out_channels=3 * n_variables,
            hidden_dim=hidden_dim,
            groups=n_variables,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            n_heads=n_heads,
            softmax_method=softmax_method,
            weight_sharing=weight_sharing,
            recurrent=recurrent,
            dropout=dropout
        )

    def forward(self, x):
        batch_size, _, sequence_length = x.size()
        x, attentions = self.attention_mechanism(x)
        n = attentions.size(2)
        x = x.reshape(batch_size, n, 3, sequence_length)
        prediction, log_v, log_beta = x.unbind(dim=2)
        log_var_aleatoric = log_beta - log_v
        return {
            'prediction': prediction,
            'attentions': attentions,
            'log_var_aleatoric': log_var_aleatoric,
            'log_v': log_v
        }

    @staticmethod
    def analysis(prediction, attentions, log_var_aleatoric, log_v):
        epistemic = torch.exp(-0.5 * log_v)

        # (bs, n, n, sequence_length)
        temp_confidence_matrix = epistemic.unsqueeze(2).expand(-1, -1, epistemic.size(1), -1)

        temp_causal_matrix, temp_external_causal_matrix = instant_attentions_to_causal_matrix(attentions)
        default_causal_matrix = weighted_mean(temp_causal_matrix, temp_confidence_matrix, dim=-1)  # (bs, n, n)

        return {
            'prediction': prediction,
            'attentions': attentions,  # (batch_size, n_heads, n, 2*n, T)
            'aleatoric': torch.exp(0.5 * log_var_aleatoric),
            'epistemic': epistemic,
            'default_causal_matrix': default_causal_matrix,
            'default_external_causal_matrix': temp_external_causal_matrix.mean(dim=-1),
            'default_confidence_matrix': temp_confidence_matrix.mean(dim=-1),
            'temp_causal_matrix': temp_causal_matrix,
            'temp_external_causal_matrix': temp_external_causal_matrix,
            'temp_confidence_matrix': temp_confidence_matrix
        }

    @staticmethod
    def loss_function(y_true, prediction, attentions, log_var_aleatoric, log_v, lambda1=0.2, beta=0.2, coeff=1e-1):
        epistemic_loss = DER_loss(y_true=y_true, y_pred=prediction, log_var=log_var_aleatoric, log_v=log_v, coeff=coeff)
        regularization = TAMCaD_regularization_loss(attentions, lambda1=lambda1, beta=beta)
        return epistemic_loss + regularization


class TemporalInstantaneousAttentionMechanism(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, groups, kernel_size, n_blocks, n_layers_per_block,
                 n_heads=1, softmax_method='softmax', dropout=0.0, weight_sharing=False, recurrent=False):
        super().__init__()
        self.n_heads = n_heads
        self.groups = groups

        self.softmax_method = get_softmax_method(softmax_method)

        # Padding for instantaneous prediction
        self.pad = nn.ConstantPad1d((0, 1), 0)

        # TCN to learn qkv projections
        self.qkv = TCN(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channels=3 * hidden_dim,  # q, k and v
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            groups=groups,
            dropout=dropout,
            weight_sharing=weight_sharing,
            recurrent=recurrent
        )
        self.fc1 = nn.Conv1d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, groups=groups)

    def forward(self, x):
        """
        Perform a forward pass through the attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, groups * dim, sequence_length).

        Returns:
            prediction (torch.Tensor): Prediction of size (batch_size, out_channels, sequence_length)
            attentions (torch.Tensor): Attention weights of size
                                      (batch_size, num_heads, groups, 2*groups, sequence_length)
        """
        batch_size, _, sequence_length = x.size()

        x = self.pad(x)

        # Apply qkv convolution and reshape: (batch_size, groups, dim, 3, sequence_length + 1)
        x = self.qkv(x).reshape(batch_size, self.groups, -1, 3, sequence_length + 1)

        # Split q, k, and v tensors: (batch_size, (2*) groups, dim, sequence_length)
        q = x[..., 0, :-1]
        k = torch.cat((x[..., 1, :-1], x[..., 1, 1:]), dim=1)
        v = torch.cat((x[..., 2, :-1], x[..., 2, 1:]), dim=1)

        # Calculate attention and predictions:
        attentions, x = instantaneous_dot_product(q, k, v, self.n_heads, self.softmax_method)
        prediction = self.fc1(x)

        return prediction, attentions


def instant_attentions_to_causal_matrix(attentions):
    """
    Convert instantaneous attention scores a causal matrix.

    Args:
        attentions (torch.Tensor): Attention scores of shape (batch_size, n_heads, n, 2*n, T),
                                  where n is the number of variables and T is the sequence length.

    Returns:
        torch.Tensor: Causal matrix representing internal dependencies with shape (batch_size, n, n, T).
        torch.Tensor: External causal matrix representing external dependencies with shape (batch_size, n, 1, T).
    """
    # Extract the number of nodes/steps from the input attentions
    n = attentions.size(2)

    # Compute mean attentions over the heads and split them into two parts: the causal matrix at t+0 and at t+1
    attn0, attn1 = attentions.mean(dim=1).chunk(2, dim=2)

    # Create an external causal matrix from the self attentions of t+1
    temp_external_causal_matrix = attn1[:, range(n), range(n), None].clone()

    # Set attentions to future self to 0 (causal masking)
    attn1[:, range(n), range(n)] = 0

    # Average the matrices to create the causal matrix
    temp_causal_matrix = (attn0 + attn1) / 2

    return temp_causal_matrix, temp_external_causal_matrix


def instantaneous_dot_product(q, k, v, n_heads, softmax_method: nn.Module, mask=None):
    """
    Perform scaled dot-product attention for temporal instantaneous attention.

    Args:
        q (torch.Tensor): Query tensor of shape (batch_size, groups, hidden_dim, sequence_length).
        k (torch.Tensor): Key tensor of shape (batch_size, 2 * groups, hidden_dim, sequence_length).
        v (torch.Tensor): Value tensor of shape (batch_size, 2 * groups, hidden_dim, sequence_length).
        n_heads (int): Number of attention heads.
        softmax_method (function): Softmax method for attention calculation.
        mask (torch.Tensor): Mask for masking out certain positions (optional).

    Returns:
        torch.Tensor: Attention weights.
        torch.Tensor: Output tensor after attention calculation.
    """
    batch_size, groups, hidden_dim, sequence_length = q.size()
    d_k = hidden_dim // n_heads

    # Reshape q, k, and v tensors
    q = q.reshape(batch_size, groups, n_heads, d_k, sequence_length).permute(0, 2, 4, 1, 3)
    k = k.reshape(batch_size, 2 * groups, n_heads, d_k, sequence_length).permute(0, 2, 4, 3, 1)
    v = v.reshape(batch_size, 2 * groups, n_heads, d_k, sequence_length).permute(0, 2, 4, 1, 3)

    # Calculate attention scores: (batch_size, num_heads, sequence_length, groups, 2 * groups)
    attention_logits = torch.matmul(q, k) * (d_k ** -0.5)

    # Apply masking if provided
    if mask is not None:
        attention_logits = attention_logits.masked_fill(mask == 0, -1e9)

    # Calculate attention weights: (batch_size, num_heads, sequence_length, groups, 2 * groups)
    attentions = softmax_method(attention_logits)

    # Calculate output projection: (batch_size, num_heads, sequence_length, groups, dk)
    output_projection = torch.matmul(attentions, v)

    # Rearrange: (batch_size, num_heads, groups, 2 * groups, sequence_length)
    attentions = attentions.permute(0, 1, 3, 4, 2)

    # Permute to original form and concatenate heads: (batch_size, groups * hidden_dim, sequence_length)
    output_projection = output_projection.permute(0, 3, 1, 4, 2).reshape(batch_size, -1, sequence_length)

    return attentions, output_projection
