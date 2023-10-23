import torch
from torch import nn

from src.models.modules.attention_activation import get_attention_activation
from src.training.result import Result
from src.models.modules.TCN import TCN


class TAMCaD(nn.Module):
    """
    Implements the Temporal Attention Mechanism for Causal Discovery (TAMCaD) model.

    Args:
        n_datasets (int): The number of datasets or time series sequences in the input.
        n_samples (int): The number of models to train for each dataset.
        n_variables (int): The number of variables in the multi-variate timeseries.
        hidden_dim (int): The dimension of the hidden state in the TAMCaD model.
        lambda1 (float): A hyperparameter controlling the regularization strength.
        beta (float): A hyperparameter controlling the regularization strength.
        dot_product (bool): A flag indicating whether to use dot product attention.
        **kwargs: Additional keyword arguments for the underlying :class:`src.models.tcn.TCN` architecture.
    """

    def __init__(self, n_datasets, n_ensembles, n_variables, hidden_dim, lambda1, beta, n_heads=None, **kwargs):
        super().__init__()
        self.groups = n_datasets * n_ensembles * n_variables

        self.lambda1 = lambda1
        self.beta = beta

        config = {"n_ensembles": n_datasets * n_ensembles, "n_variables": n_variables, "hidden_dim": hidden_dim}

        if n_heads:
            self.tamcad = ScaledDotProductTAMCaD(**config, **kwargs, n_heads=n_heads)
        else:
            self.tamcad = SimpleTAMCaD(**config, **kwargs)

        reg_mask = None
        if beta and n_ensembles > 1:
            n_var_instant = 2 * n_variables if self.tamcad.instantaneous else n_variables
            reg_mask = torch.rand(n_ensembles, n_variables, n_var_instant, 1) < 0.5
        self.register_buffer("reg_mask", reg_mask)

        self.convert_instant = ConvertInstant(n_variables) if self.tamcad.instantaneous else None

        self.prediction = nn.Sequential(
            nn.Conv1d(in_channels=self.groups * hidden_dim,
                      out_channels=self.groups * hidden_dim // 2,
                      kernel_size=1, groups=self.groups),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.groups * hidden_dim // 2,
                      out_channels=self.groups,
                      kernel_size=1, groups=self.groups)
        )

    def forward(self, x, temperature=1.0):
        """
        Forward pass through the TAMCaD model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_datasets, n_ensembles, n_variables * 3, sequence_length).

        Returns:
            dict: Dictionary containing the prediction and attentions.
        """
        batch_size, n_datasets, n_ensembles, n_var, sequence_length = x.size()
        groups = (n_datasets, n_ensembles, n_var // 3)

        x = x.reshape(batch_size, -1, sequence_length)

        # context: (batch_size, n_datasets * n_samples * n_var * dim, sequence_length)
        # attentions: (batch_size, n_datasets * n_samples, n_var, (2*) n_var, sequence_length)
        # prediction: (batch_size, n_datasets * n_samples * n_var, sequence_length)
        context, attentions = self.tamcad(x, temperature=temperature)
        prediction = self.prediction(context)

        # attentions: (batch_size, n_datasets, n_samples, n_var, (2*) n_var, sequence_length)
        # prediction: (batch_size, n_datasets, n_samples, n_var, sequence_length)
        prediction = prediction.reshape(batch_size, *groups, -1)
        attentions = attentions.reshape(batch_size, *groups, -1, prediction.size(-1))

        return {
            'prediction': prediction,
            'attentions': attentions
        }

    def compute_loss(self, y_true, prediction, attentions):
        loss = ((prediction - y_true) ** 2).mean()  # MSE loss

        if self.tamcad.instantaneous and self.lambda1:
            n = self.tamcad.n_variables
            loss = loss + self.lambda1 * attentions[..., range(n), range(n, 2 * n), :].mean()

        if self.reg_mask is not None:
            print(self.reg_mask.size())
            loss = loss + self.beta * torch.masked_fill(attentions, mask=self.reg_mask, value=0).mean()

        return loss

    def analysis(self, prediction, attentions):
        """
        prediction: (batch_size, n_datasets, n_samples, n, T)
        attentions: (batch_size, n_datasets, n_samples, n, (2*)n, T)
        context: (batch_size, n_datasets, n_samples, n, dim, T)
        """
        if self.convert_instant is not None:
            # attentions: (batch_size, n_datasets, n_samples, n, n, T)
            attentions = self.convert_instant(attentions)
        attentions_mean = attentions.mean(dim=-1)  # (batch_size, n_datasets, n_samples, n, n)
        return Result(
            prediction=prediction.mean(dim=2),  # (batch_size, n_datasets, n, T)
            causal_matrix=attentions_mean.mean(dim=2),  # (batch_size, n_datasets, n, n)
            causal_matrix_std=attentions_mean.std(dim=2),  # (batch_size, n_datasets, n, n)
            temporal_causal_matrix=attentions.mean(dim=2),  # (batch_size, n_datasets, n, n, T)
            temporal_causal_matrix_std=attentions.std(dim=2)  # (batch_size, n_datasets, n, n, T)
        )


class SimpleTAMCaD(nn.Module):
    def __init__(self, n_ensembles, n_variables, hidden_dim, attention_activation, instantaneous, **kwargs):
        super().__init__()
        self.n_ensembles = n_ensembles
        self.n_variables = n_variables
        self.hidden_dim = hidden_dim
        self.instantaneous = instantaneous

        # Padding for instantaneous prediction
        self.pad = nn.ConstantPad1d((0, 1), 0) if instantaneous else None

        self.attention_mechanism = SimpleAttention(attention_activation)

        # TCN to learn the projections
        groups = n_ensembles * n_variables
        channels = groups * (hidden_dim + n_variables)
        if instantaneous:
            channels *= 2

        self.tcn = TCN(
            in_channels=groups * 3,
            hidden_dim=groups * hidden_dim,
            out_channels=channels,
            groups=groups,
            **kwargs
        )

    def forward(self, x, temperature=1.0):
        """
        Perform a forward pass through the attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_ensembles * n_var * dim, sequence_length).
            temperature (float): softmax temperature

        Returns:
            embeddings (torch.Tensor): Prediction of size (batch_size, n_ensembles * n_var * dim, sequence_length).
            attentions (torch.Tensor): Attention weights of size (batch_size, n_ensembles, n_var, n_var, sequence_length).
        """
        batch_size, _, sequence_length = x.size()

        if self.instantaneous:
            x = self.pad(x)
            # Apply qkv convolution and reshape: (batch_size * k, 2, n_var + hidden_dim, sequence_length + 1)
            x = self.tcn(x)
            x = x.reshape(-1, self.n_variables, 2, self.n_variables + self.hidden_dim, x.size(-1))
            # (batch_size * k, n_var + hidden_dim, sequence_length)
            x, x_instant = x[..., 0, :, :-1], x[..., 1, :, 1:]
        else:
            # Apply qkv convolution and reshape: (batch_size * groups, n_var + hidden_dim, sequence_length)
            x = self.tcn(x)
            x = x.reshape(-1, self.n_variables, self.n_variables + self.hidden_dim, x.size(-1))
            x_instant = None

        # Calculate attention and predictions:
        x, attentions = self.attention_mechanism(x, x_instant, temperature=temperature)

        x = x.reshape(batch_size, -1, x.size(-1))
        attentions = attentions.reshape(batch_size, self.n_ensembles, self.n_variables, -1, x.size(-1))

        return x, attentions


class SimpleAttention(nn.Module):
    def __init__(self, attention_activation: str, mask=None):
        super().__init__()
        self.mask = mask
        self.attention_activation = get_attention_activation(attention_activation)

    def forward(self, x, x_instant=None, temperature=1.0):
        """
        Perform scaled dot-product attention for temporal instantaneous attention.

        Args:
            x (torch.Tensor): Context tensor of shape (batch_size, n_var, (hidden_dim + n_var), sequence_length).
            x_instant (torch.Tensor): Context tensor of shape (batch_size, n_var, (hidden_dim + n_var), sequence_length).
            temperature (float): softmax temperature

        Returns:
            torch.Tensor: Output tensor after attention calculation of size (batch_size, n_var * hidden_dim, sequence_length).
            torch.Tensor: Attention weights of size (batch_size, n_var, (2*) n_var, sequence_length).
        """
        batch_size, n_var, _, sequence_length = x.size()

        # (batch_size, n_var, hidden_dim, sequence_length), (batch_size, n_var, n_var, sequence_length)
        context, attentions = x[..., n_var:, :], x[..., :n_var, :]

        if x_instant is not None:
            # (batch_size, n_var, hidden_dim, sequence_length), (batch_size, n_var, n_var, sequence_length)
            context_instant, attentions_instant = x_instant[..., n_var:, :], x_instant[..., :n_var, :]
            context = torch.cat((context, context_instant), dim=1)
            attentions = torch.cat((attentions, attentions_instant), dim=2)

        # Apply masking if provided
        if self.mask is not None:
            attentions = attentions.masked_fill(self.mask == 0, -1e9)

        attentions = self.attention_activation(attentions / temperature, dim=2)

        # x: (batch_size, n_var * hidden_dim, sequence_length)
        x = torch.einsum('bijt, bjdt -> bidt', attentions, context).reshape(batch_size, -1, sequence_length)

        # attentions: (batch_size, n_var, (2*) n_var, sequence_length)
        attentions = attentions.reshape(batch_size, n_var, -1, sequence_length)

        return x, attentions


class ScaledDotProductTAMCaD(nn.Module):
    def __init__(self, n_ensembles, n_variables, hidden_dim, attention_activation, n_heads, instantaneous, **kwargs):
        super().__init__()
        self.n_ensembles = n_ensembles
        self.n_variables = n_variables
        self.hidden_dim = hidden_dim
        self.instantaneous = instantaneous

        # Padding for instantaneous prediction
        self.pad = nn.ConstantPad1d((0, 1), 0) if instantaneous else None

        self.attention_mechanism = ScaledDotProductTemporalAttention(n_heads, attention_activation)

        # TCN to learn qkv projections
        groups = n_ensembles * n_variables
        qkv = 5 if instantaneous else 3
        self.tcn = TCN(
            in_channels=groups * 3,
            hidden_dim=groups * hidden_dim,
            out_channels=groups * hidden_dim * qkv,
            groups=groups,
            **kwargs
        )

    def forward(self, x, temperature=1.0):
        """
        Perform a forward pass through the attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, k * n_var * dim, sequence_length).
            temperature (float): softmax temperature

        Returns:
            embeddings (torch.Tensor): Prediction of size (batch_size, k * n_var * dim, sequence_length).
            attentions (torch.Tensor): Attention weights of size (batch_size, k, n_var, n_var, sequence_length).
        """
        batch_size = x.size(0)

        if self.instantaneous:
            x = self.pad(x)

            # Apply qkv convolution and reshape: batch_size * k * n_var, 5, hidden_dim, sequence_length + 1)
            x = self.tcn(x)
            x = x.reshape(-1, self.n_variables, 5, self.hidden_dim, x.size(-1))

            # Split q, k, and v tensors: (batch_size * k * n_var, hidden_dim, sequence_length)
            q, k, v = x[..., :3, :, :-1].unbind(dim=-3)
            k_instant, v_instant = x[..., 3:, :, 1:].unbind(dim=-3)
        else:
            # Apply qkv convolution and reshape: (batch_size * k * n_var, 3, hidden_dim, sequence_length)
            x = self.tcn(x)
            x = x.reshape(-1, self.n_variables, 3, self.hidden_dim, x.size(-1))

            # Split q, k, and v tensors: (batch_size * k * n_var, hidden_dim, sequence_length)
            q, k, v = x.unbind(dim=-3)
            k_instant, v_instant = None, None

        # Calculate attention and predictions:
        x, attentions = self.attention_mechanism(q, k, v, k_instant, v_instant, temperature=temperature)

        x = x.reshape(batch_size, -1, x.size(-1))
        attentions = attentions.reshape(batch_size, self.n_ensembles, self.n_variables, -1, x.size(-1))

        return x, attentions


class ScaledDotProductTemporalAttention(nn.Module):
    def __init__(self, n_heads: int, attention_activation: str, mask=None):
        super().__init__()
        self.mask = mask
        self.n_heads = n_heads
        self.attention_activation = get_attention_activation(attention_activation)

    def forward(self, q, k, v, k_instant=None, v_instant=None, temperature=1.0):
        """
        Perform scaled dot-product attention for temporal (instantaneous0 attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, groups, hidden_dim, sequence_length).
            k (torch.Tensor): Key tensor of shape (batch_size, groups, hidden_dim, sequence_length).
            v (torch.Tensor): Value tensor of shape (batch_size, groups, hidden_dim, sequence_length).
            k_instant (torch.Tensor): Value tensor of shape (batch_size, groups, hidden_dim, sequence_length).
            v_instant (torch.Tensor): Value tensor of shape (batch_size, groups, hidden_dim, sequence_length).
            temperature (float): softmax temperature

        Returns:
            torch.Tensor: Output tensor after attention calculation of size (batch_size, n_var * hidden_dim, sequence_length).
            torch.Tensor: Attention weights of size (batch_size, n_var, n_var, sequence_length).
        """
        if k_instant is not None:
            k = torch.cat((k, k_instant), dim=1)
            v = torch.cat((v, v_instant), dim=1)

        batch_size, groups_q, hidden_dim, sequence_length = q.size()
        groups_kv = k.size(1)
        d_k = hidden_dim // self.n_heads

        # Reshape q, k, and v tensors
        # (batch_size, n_heads, seq_length, groups_q, dk)
        q = q.reshape(batch_size, groups_q, self.n_heads, d_k, sequence_length).permute(0, 2, 4, 1, 3)
        # (batch_size, n_heads, seq_length, dk, groups_kv)
        k = k.reshape(batch_size, groups_kv, self.n_heads, d_k, sequence_length).permute(0, 2, 4, 3, 1)
        # (batch_size, n_heads, seq_length, groups_kv, dk)
        v = v.reshape(batch_size, groups_kv, self.n_heads, d_k, sequence_length).permute(0, 2, 4, 1, 3)

        # Calculate attention scores: (batch_size, num_heads, sequence_length, groups_q, groups_kv)
        attention_logits = torch.matmul(q, k) * ((d_k ** -0.5) / temperature)

        # Apply masking if provided
        if self.mask is not None:
            attention_logits = attention_logits.masked_fill(self.mask == 0, -1e9)

        # Calculate attention weights: (batch_size, num_heads, sequence_length, groups_q, groups_kv)
        attentions = self.attention_activation(attention_logits, dim=-1)

        # Calculate output projection: (batch_size, num_heads, sequence_length, groups_q, dk)
        x = torch.matmul(attentions, v)

        # Rearrange: (batch_size, num_heads, groups_q, groups_kv, sequence_length)
        attentions = attentions.permute(0, 1, 3, 4, 2)
        # Mean over heads: (batch_size, groups_q, groups_kv, sequence_length)
        attentions = attentions.mean(dim=1)

        # Permute to original form and concatenate heads: (batch_size, groups_q * hidden_dim, sequence_length)
        x = x.permute(0, 3, 1, 4, 2).reshape(batch_size, -1, sequence_length)

        return x, attentions


class ConvertInstant(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.unflatten = torch.nn.Unflatten(-2, (2, -1))
        mask = torch.zeros(groups, 2, groups, 1, dtype=torch.bool)
        mask[range(groups), 1, range(groups)] = True
        self.register_buffer('mask', mask)

    def forward(self, attentions):
        attentions = self.unflatten(attentions)
        attentions = torch.masked_fill(attentions, self.mask, value=0)
        return attentions.sum(dim=-3)


