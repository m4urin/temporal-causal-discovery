import torch
from torch import nn

from src.models.softmax import get_softmax_method


class SimpleAttention(nn.Module):
    def __init__(self, n_variables: int, softmax_method: str, instantaneous: bool, mask=None):
        super().__init__()
        self.mask = mask
        self.n_variables = n_variables
        self.softmax_method = get_softmax_method(softmax_method)
        self.convert_instant = ConvertInstant(n_variables) if instantaneous else None

    def forward(self, x, x_instant=None, temperature=1.0):
        """
        Perform scaled dot-product attention for temporal instantaneous attention.

        Args:
            x (torch.Tensor): Context tensor of shape (batch_size, n_var * (hidden_dim + n_var), sequence_length).
            x_instant (torch.Tensor): Context tensor of shape (batch_size, n_var * (hidden_dim + n_var), sequence_length).
            temperature (float): softmax temperature

        Returns:
            torch.Tensor: Attention weights of size (batch_size, n_var, n_var, sequence_length).
            torch.Tensor: Output tensor after attention calculation
            of size (batch_size, n_var * hidden_dim, sequence_length).
        """
        batch_size, _, sequence_length = x.size()

        # (batch_size, n_var, hidden_dim, sequence_length), (batch_size, n_var, n_var, sequence_length)
        context, attentions = self.split_context_attention(x)

        if x_instant is not None:
            # (batch_size, n_var, hidden_dim, sequence_length), (batch_size, n_var, n_var, sequence_length)
            context_instant, attentions_instant = self.split_context_attention(x_instant)
            context = torch.cat((context, context_instant), dim=-3)
            attentions = torch.cat((attentions, attentions_instant), dim=-2)

        # Apply masking if provided
        if self.mask is not None:
            attentions = attentions.masked_fill(self.mask == 0, -1e9)

        attentions = self.softmax_method(attentions / temperature, dim=-2)

        # x: (batch_size, n_var, hidden_dim, sequence_length)
        x = torch.einsum('bijt, bjdt -> bidt', attentions, context)

        # x: (batch_size, n_var * hidden_dim, sequence_length)
        x = x.reshape(batch_size, -1, sequence_length)

        # attentions: (batch_size, n_var, (2*) n_var, sequence_length)
        attentions = attentions.reshape(batch_size, self.n_variables, -1, sequence_length)

        # attentions: (batch_size, n_var, n_var, sequence_length)
        if self.convert_instant is not None:
            attentions = self.convert_instant(attentions)

        return x, attentions

    def split_context_attention(self, x):
        batch_size, _, sequence_length = x.size()

        # (batch_size, n_variables, hidden_dim + n_variables, sequence_length)
        x = x.reshape(batch_size, self.n_variables, -1, sequence_length)

        # (batch_size, n_variables, n_variables, sequence_length)
        attentions = x[..., :self.n_variables, :]

        # (batch_size, n_variables, hidden_dim, sequence_length)
        context = x[..., self.n_variables:, :]

        return context, attentions


class ScaledDotProductAttention(nn.Module):
    """
    Perform scaled dot-product attention for temporal instantaneous attention.

    Args:
        n_heads (int): Number of attention heads.
        softmax_method (function): Softmax method for attention calculation.
        mask (torch.Tensor): Mask for masking out certain positions (optional).
    """
    def __init__(self, n_variables: int, n_heads: int, softmax_method: str, instantaneous: bool, mask=None):
        super().__init__()
        self.mask = mask
        self.n_heads = n_heads
        self.softmax_method = get_softmax_method(softmax_method)
        self.convert_instant = ConvertInstant(n_variables) if instantaneous else None

    def forward(self, q, k, v, temperature=1.0):
        """
        Perform scaled dot-product attention for temporal instantaneous attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, groups, hidden_dim, sequence_length).
            k (torch.Tensor): Key tensor of shape (batch_size, groups (* 2), hidden_dim, sequence_length).
            v (torch.Tensor): Value tensor of shape (batch_size, groups (* 2), hidden_dim, sequence_length).
            temperature (float): softmax temperature

        Returns:
            torch.Tensor: Attention weights.
            torch.Tensor: Output tensor after attention calculation.
        """
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
        attentions = self.softmax_method(attention_logits, dim=-1)

        # Calculate output projection: (batch_size, num_heads, sequence_length, groups_q, dk)
        x = torch.matmul(attentions, v)

        # Rearrange: (batch_size, num_heads, groups_q, groups_kv, sequence_length)
        attentions = attentions.permute(0, 1, 3, 4, 2)

        # Permute to original form and concatenate heads: (batch_size, groups_q * hidden_dim, sequence_length)
        x = x.permute(0, 3, 1, 4, 2).reshape(batch_size, -1, sequence_length)

        if self.convert_instant is not None:
            attentions = self.convert_instant(attentions)

        return x, attentions


class ConvertInstant(nn.Module):
    def __init__(self, n_variables):
        super().__init__()
        mask = torch.zeros(n_variables, 2 * n_variables, 1, dtype=torch.bool)
        mask[range(n_variables), range(n_variables, 2 * n_variables)] = True
        self.register_buffer('mask', mask)

    def forward(self, attentions):
        a1, a2 = torch.masked_fill(attentions, self.mask, value=0).chunk(2, dim=-2)
        return a1 + a2
