import torch
from torch import nn


class TemporalInstantaneousAttention(nn.Module):
    def __init__(self, hidden_dim, out_channels, groups, num_heads, softmax_method: str):
        super().__init__()
        self.groups = groups
        self.num_heads = num_heads

        if softmax_method == 'softmax':
            self.softmax_method = f_softmax
        if softmax_method == 'softmax-1':
            self.softmax_method = f_softmax_1
        if softmax_method == 'normalized_sigmoid':
            self.softmax_method = f_normalized_sigmoid
        else:
            raise NotImplementedError(f"Method '{softmax_method}' is not a valid method.")

        self.qkv = nn.Conv1d(in_channels=hidden_dim, out_channels=3 * hidden_dim, kernel_size=1, groups=groups)
        self.prediction = nn.Conv1d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, groups=groups)
        self.pad = nn.ConstantPad1d((0, 1), 0)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Reset the parameters using Xavier initialization.
        """
        nn.init.xavier_uniform_(self.qkv.weight)
        self.qkv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.prediction.weight)
        self.prediction.bias.data.fill_(0)

    def forward(self, x):
        batch_size, hidden_dim, sequence_length = x.size()
        dim = hidden_dim // self.groups

        x = self.pad(x)

        # (batch_size, groups, dim, 3, sequence_length + 1)
        x = self.qkv(x).reshape(batch_size, self.groups, dim, 3, sequence_length + 1)

        # (batch_size, groups, dim, sequence_length)
        q = x[..., 0, :-1]
        # (batch_size, 2 * groups, dim, sequence_length)
        k = torch.cat((x[..., 1, :-1], x[..., 1, 1:]), dim=1)
        v = torch.cat((x[..., 2, :-1], x[..., 2, 1:]), dim=1)

        # (batch_size, num_heads, groups, 2*groups, sequence_length), (batch_size, hidden_dim, sequence_length)
        attention, x = temporal_instantaneous_scaled_dot_product(q, k, v, self.num_heads, self.softmax_method)

        x = self.prediction(x)

        return x, attention


def temporal_instantaneous_scaled_dot_product(q, k, v, num_heads, softmax_method, mask=None):
    # q is of size (batch_size, groups, hidden_dim, sequence_length)
    # k and v are of size (batch_size, 2 * groups, hidden_dim, sequence_length)
    batch_size, groups, hidden_dim, sequence_length = q.size()
    d_k = hidden_dim // num_heads

    # (batch_size, 2 * groups, dim, sequence_length)

    # Split into multiple heads: (batch_size, num_heads, sequence_length, groups, dk)
    q = q.reshape(batch_size, groups, num_heads, d_k, sequence_length).permute(0, 2, 4, 1, 3)
    k = k.reshape(batch_size, 2 * groups, num_heads, d_k, sequence_length).permute(0, 2, 4, 3, 1)  # transposed
    v = v.reshape(batch_size, 2 * groups, num_heads, d_k, sequence_length).permute(0, 2, 4, 1, 3)

    # Scaled dot-product attention: (batch_size, num_heads, sequence_length, groups, 2 * groups)
    scores = torch.matmul(q, k) * (d_k ** -0.5)

    # Masking
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Calculate attention weights: (batch_size, num_heads, sequence_length, groups, 2 * groups)
    attention_weights = softmax_method(scores)

    # Multiply attention weights with v to get the output: (batch_size, num_heads, sequence_length, groups, dk)
    output_projection = torch.matmul(attention_weights, v)

    # Permute to original form: (batch_size, num_heads, groups, 2 * groups, sequence_length)
    attention_weights = attention_weights.permute(0, 1, 3, 4, 2)

    # Permute to original form and concatenate heads: (batch_size, groups * hidden_dim, sequence_length)
    output_projection = output_projection.permute(0, 3, 1, 4, 2).reshape(batch_size, -1, sequence_length)

    return attention_weights, output_projection


def f_softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)


def f_softmax_1(x, dim=-1):
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    return x_exp / (torch.exp(-maxes) + torch.sum(x_exp, dim, keepdim=True))


def f_normalized_sigmoid(x, dim=-1):
    z = torch.sigmoid(x)
    return z / (1e-9 + torch.sum(z, dim=dim, keepdim=True))  # x.size(dim)

