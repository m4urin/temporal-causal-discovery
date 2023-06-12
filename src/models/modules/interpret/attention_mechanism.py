import math
import torch
from torch import nn

from src.models.model_outputs import ModelOutput
from src.models.modules.interpret.interpret_module import InterpretModule
from src.models.modules.temporal.temporal_variational_layer import TemporalVariationalLayer


def scaled_dot_product(q, k, v, mask=None):
    """
    Applies scaled dot product attention mechanism.

    Args:
        q (torch.Tensor): Query tensor of shape (..., sequence_length, embed_dim).
        k (torch.Tensor): Key tensor of shape (..., sequence_length, embed_dim).
        v (torch.Tensor): Value tensor of shape (..., sequence_length, embed_dim).
        mask (torch.Tensor, optional): Mask tensor of shape (..., sequence_length, sequence_length)
            indicating which positions should be masked. Default is None.

    Returns:
        attention (torch.Tensor): Attention tensor of shape (..., sequence_length, sequence_length).
        values (torch.Tensor): Output tensor of shape (..., sequence_length, embed_dim).
    """
    d_k = q.size()[-1]

    # Compute attention logits
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))

    # Compute attention weights
    attention = torch.softmax(attn_logits, dim=-1)

    # Compute output values
    values = torch.matmul(attention, v)

    return attention, values


class AttentionDefault(InterpretModule):
    def __init__(self, in_channels: int, groups: int, num_external_variables: int, use_variational_layer: bool):
        """
        Initializes the AttentionInstant layer.

        Args:
            in_channels (int): Number of input channels.
            groups (int): Number of groups.
            num_external_variables (int): Number of external variables.
            use_variational_layer (bool): Flag indicating whether to use a variational layer.
        """
        super().__init__(
            in_channels,
            groups,
            num_external_variables,
            use_variational_layer,
            use_instantaneous_predictions=False
        )
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.kv_proj = nn.Linear(self.dim, 2 * self.dim)
        self.o_proj = nn.Linear(self.dim, self.dim)

        if use_variational_layer:
            self.predict = nn.Sequential(
                nn.ReLU(),
                TemporalVariationalLayer(self.n * self.dim, self.n, groups=self.n)
            )
        else:
            self.predict = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(self.n * self.dim, self.n, kernel_size=1, groups=self.n)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Reset the parameters using Xavier initialization.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x) -> ModelOutput:
        """
        Forward pass of the temporal attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_len).

        Returns:
             ModelOutput: Object containing output tensors and attention tensors.
                - 'x' (torch.Tensor): Output tensor of shape (batch_size, out_channels, seq_len).
                - 'attn' (torch.Tensor): Attention tensor of shape (batch_size, groups, groups, seq_len).
                - 'attn_external_variables' (torch.Tensor, optional): Attention tensor for external variables of
                   shape (batch_size, groups, groups, seq_len).
                - 'mu' (torch.Tensor, optional): Mean tensor for variational layer of shape (batch_size, n, seq_len).
                - 'std' (torch.Tensor, optional): Standard deviation tensor for variational layer
                   of shape (batch_size, n, seq_len).
                - 'loss' (torch.Tensor, optional): KL divergence loss tensor for variational layer of shape ().
        """
        batch_size, _, seq_len = x.size()

        # (batch_size, seq_len, groups, dim)
        x = x.transpose(-2, -1).reshape(batch_size, seq_len, self.groups, self.dim)

        # embeddings at timestep t without external variables
        q = self.q_proj(x[:, :, :self.n])  # (batch_size, seq_len, n, dim)

        k, v = self.kv_proj(x).chunk(2, dim=-1)  # (batch_size, seq_len, groups, dim)

        # attn: (batch_size, seq_len, n, groups)
        # x: (batch_size, seq_len, n, dim)
        attn, x = scaled_dot_product(q, k, v)

        # Apply output projection
        x = self.o_proj(x)  # (batch_size, seq_len, n, dim)

        # Reshape attention tensor
        attn = attn.permute(0, 3, 2, 1)  # (batch_size, groups, n, seq_len)

        result = {'attn': attn[:, :self.n]}
        if self.e > 0:
            result['attn_external_variables'] = attn[:, self.n:]

        x = x.reshape(batch_size, seq_len, -1).transpose(-1, -2)  # (batch_size, n * dim, seq_len)

        if self.use_variational_layer:
            x, mu, std, kl_loss = self.predict(x)  # (batch_size, n, seq_len)
            return ModelOutput(x_pred=x, mu_pred=mu, std_pred=std, kl_loss=kl_loss,
                               uncertainty_method='variational', **result)
        else:
            x = self.predict(x)  # (batch_size, n, seq_len)
            return ModelOutput(x_pred=x, **result)


class AttentionInstant(InterpretModule):
    def __init__(self, in_channels: int, groups: int, num_external_variables: int, use_variational_layer: bool):
        """
        Initializes the AttentionInstant layer.

        Args:
            in_channels (int): Number of input channels.
            groups (int): Number of groups.
            num_external_variables (int): Number of external variables.
            use_variational_layer (bool): Flag indicating whether to use a variational layer.
        """
        super().__init__(
            in_channels,
            groups,
            num_external_variables,
            use_variational_layer,
            use_instantaneous_predictions=True
        )
        self.register_buffer('mask', self._create_mask())

        self.pad = nn.ConstantPad1d((0, 1), 0)

        self.q_proj = nn.Linear(self.dim, self.dim)
        self.kv_proj = nn.Linear(self.dim, 2 * self.dim)
        self.o_proj = nn.Linear(self.dim, self.dim)

        if use_variational_layer:
            self.predict = nn.Sequential(
                nn.ReLU(),
                TemporalVariationalLayer(self.n * self.dim, self.n, groups=self.n)
            )
        else:
            self.predict = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(self.n * self.dim, self.n, kernel_size=1, groups=self.n)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Reset the parameters using Xavier initialization.
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def _create_mask(self):
        """
        Creates a mask tensor.

        Returns:
            torch.Tensor: Mask tensor of shape (1, 1, n, 2*n+e).
        """
        mask = torch.ones(self.n, self.groups + self.n, dtype=torch.int)
        mask[:, self.groups:] = 1 - torch.eye(self.n)
        mask = mask[None, None, ...]  # (1, 1, n, 2*n+e)
        return mask

    def forward(self, x) -> ModelOutput:
        """
        Forward pass of the temporal attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_len).

        Returns:
            ModelOutput: Object containing output tensors and attention tensors.
                - 'x' (torch.Tensor): Output tensor of shape (batch_size, out_channels, seq_len).
                - 'attn' (torch.Tensor): Attention tensor of shape (batch_size, groups, groups, seq_len).
                - 'attn_instantaneous' (torch.Tensor): Instantaneous attention tensor
                   of shape (batch_size, groups, groups, seq_len).
                - 'attn_external_variables' (torch.Tensor, optional): Attention tensor for external variables
                   of shape (batch_size, groups, groups, seq_len).
                - 'mu' (torch.Tensor, optional): Mean tensor for variational layer of shape (batch_size, n, seq_len).
                - 'std' (torch.Tensor, optional): Standard deviation tensor for variational layer
                   of shape (batch_size, n, seq_len).
                - 'loss' (torch.Tensor, optional): KL divergence loss tensor for variational layer of shape ().
        """
        batch_size, _, seq_len = x.size()

        x = self.pad(x)

        # (batch_size, seq_len+1, groups, dim)
        x = x.transpose(-2, -1).reshape(batch_size, seq_len + 1, self.groups, self.dim)

        # embeddings at timestep t without external variables
        q = self.q_proj(x[:, :-1, :self.n])  # (batch_size, seq_len, n, dim)

        # embeddings at timestep t including external variables and
        # embeddings at timestep t+1 without external embeddings
        kv = torch.cat((x[:, :-1], x[:, 1:, :self.n]), dim=2)  # (batch_size, seq_len, n+e+n, dim)

        k, v = self.kv_proj(kv).chunk(2, dim=-1)  # (batch_size, seq_len, n+e+n, dim)

        # attn: (batch_size, seq_len, n, n+e+n)
        # values: (batch_size, seq_len, n, dim)
        attn, x = scaled_dot_product(q, k, v, mask=self.mask)

        # Apply output projection
        x = self.o_proj(x)  # (batch_size, seq_len, n, dim)

        # Reshape attention tensor
        attn = attn.permute(0, 3, 2, 1)  # (batch_size, n+e+n, n, seq_len)

        result = {
            'attn': attn[:, :self.n],
            'attn_instantaneous': attn[:, self.groups:]
        }
        if self.e > 0:
            result['attn_external_variables'] = attn[:, self.n:self.groups]

        x = x.reshape(batch_size, seq_len, -1).transpose(-1, -2)  # (batch_size, n * dim, seq_len)

        if self.use_variational_layer:
            x, mu, std, kl_loss = self.predict(x)  # (batch_size, n, seq_len)
            return ModelOutput(x_pred=x, mu_pred=mu, std_pred=std, kl_loss=kl_loss,
                               uncertainty_method='variational', **result)
        else:
            x = self.predict(x)  # (batch_size, n, seq_len)
            return ModelOutput(x_pred=x, **result)
