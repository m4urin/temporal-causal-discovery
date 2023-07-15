import torch
import torch.nn as nn

from src.models.model_outputs import ModelOutput
from src.models.modules.interpret.interpret_module import InterpretModule
from src.models.modules.temporal.temporal_variational_layer import TemporalVariationalLayer


def sum_of_distributions(mu: torch.Tensor, std: torch.Tensor, dim: int, keep_dim: bool = False) -> tuple:
    """
    Computes the sum of distributions along a given dimension.

    Args:
        mu (torch.Tensor): Mean tensor.
        std (torch.Tensor): Standard deviation tensor.
        dim (int): Dimension along which to compute the sum.
        keep_dim (bool, optional): Whether to keep the dimension after summing. Defaults to False.

    Returns:
        tuple: Tuple containing the summed mean and standard deviation tensors.
    """
    mu_hat = mu.sum(dim=dim, keepdim=keep_dim)
    var_hat = std.pow(2).sum(dim=dim, keepdim=keep_dim)
    std_hat = torch.sqrt(var_hat)
    return mu_hat, std_hat


def attn_regularization(attn: torch.Tensor, attn_instantaneous: torch.Tensor = None):
    attn_loss = attn.abs().mean(dim=(1, 2, 3))  # (batch_size,)
    if attn_instantaneous is not None:
        attn_loss += attn_instantaneous.abs().mean(dim=(1, 2, 3))  # (batch_size,)

    attn_loss *= attn.size(1)  # scale to number of variables
    return attn_loss


class NavarDefault(InterpretModule):
    """
    Default implementation of Navar interpretation layer.

    Args:
        in_channels (int): Number of input channels.
        groups (int): Number of groups in the model.
        num_external_variables (int): Number of external variables.
    """

    def __init__(self, in_channels: int, groups: int, num_external_variables: int):
        super().__init__(in_channels, groups, num_external_variables)
        self.contributions = nn.Conv1d(in_channels, groups * self.n, kernel_size=1, groups=groups, bias=False)
        self.biases = nn.Parameter(torch.empty(1, self.n, 1))
        self.biases.data.fill_(0.0001)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Performs a forward pass through the NavarDefault interpretation layer.

        Args:
            x (torch.Tensor): Input tensor of size (batch_size, in_channels, sequence_length).

        Returns:
            ModelOutput: Object containing the output tensors: 'x', 'attn', and 'attn_external_variables'.
        """
        batch_size, _, seq_len = x.size()

        x = self.contributions(x)

        x = x.reshape(batch_size, self.groups, self.n, seq_len)

        causal_matrix = x.std(dim=-1).permute(0, 2, 1).detach()

        attn = x[:, :self.n].permute(0, 2, 1, 3)
        attn_ext = x[:, self.n:].permute(0, 2, 1, 3)

        x = x.sum(dim=1) + self.biases  # (batch_size, n, seq_len)

        return ModelOutput(x_pred=x, attn=attn, attn_external_variables=attn_ext,
                           attn_loss=attn_regularization(attn), causal_matrix=causal_matrix)


class NavarVar(InterpretModule):
    """
    Variation of Navar interpretation layer with a variational layer.

    Args:
        in_channels (int): Number of input channels.
        groups (int): Number of groups in the model.
        num_external_variables (int): Number of external variables.
    """

    def __init__(self, in_channels: int, groups: int, num_external_variables: int):
        super().__init__(in_channels, groups, num_external_variables, use_variational_layer=True)
        self.contributions = TemporalVariationalLayer(in_channels, groups * self.n, groups=groups)
        self.biases = nn.Parameter(torch.empty(1, self.n, 1))
        self.biases.data.fill_(0.0001)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Performs a forward pass through the NavarVar interpretation layer.

        Args:
            x (torch.Tensor): Input tensor of size (batch_size, in_channels, sequence_length).

        Returns:
            ModelOutput: Object containing the output tensors: 'x', 'mu', 'std', 'attn', 'attn_external_variables', and 'loss'.
        """
        batch_size, _, seq_len = x.size()

        x, mu, std, kl_loss = self.contributions(x)

        x = x.reshape(batch_size, self.groups, self.n, seq_len)
        mu = mu.reshape(batch_size, self.groups, self.n, seq_len)
        std = std.reshape(batch_size, self.groups, self.n, seq_len)

        causal_matrix = x.std(dim=-1).permute(0, 2, 1).detach()

        attn = x[:, :self.n].permute(0, 2, 1, 3)
        attn_ext = x[:, self.n:].permute(0, 2, 1, 3)

        mu, std = sum_of_distributions(mu, std, dim=1)  # (batch_size, n, seq_len)

        x = x.sum(dim=1)  # (batch_size, n, seq_len)

        return ModelOutput(
            x_pred=x + self.biases,
            mu_pred=mu + self.biases,
            std_pred=std,
            attn=attn,
            attn_external_variables=attn_ext,
            kl_loss=kl_loss,
            attn_loss=attn_regularization(attn),
            causal_matrix=causal_matrix
        )


class NavarInstant(InterpretModule):
    """
    Variation of Navar interpretation layer with instantaneous predictions.

    Args:
        in_channels (int): Number of input channels.
        groups (int): Number of groups in the model.
        num_external_variables (int): Number of external variables.
    """

    def __init__(self, in_channels: int, groups: int, num_external_variables: int):
        super().__init__(in_channels, groups, num_external_variables, use_instantaneous_predictions=True)
        self.contributions_t0 = nn.Conv1d(in_channels, groups * self.n, kernel_size=1, groups=groups, bias=False)
        self.contributions_t1 = nn.Conv1d(self.n * self.dim, self.n * self.n, kernel_size=1, groups=self.n, bias=False)

        self.biases = nn.Parameter(torch.empty(1, self.n, 1))
        self.biases.data.fill_(0.0001)

        mask = 1 - torch.eye(self.n, self.n, dtype=torch.int).reshape(1, self.n, self.n, 1)
        self.register_buffer('mask', mask)

        self.pad = nn.ConstantPad1d((0, 1), 0)

    def calc(self, x: torch.Tensor, layer, mask=None):
        """
        Calculates the attention and prediction using a given layer.

        Args:
            x (torch.Tensor): Input tensor.
            layer: The layer to use for calculation.
            mask: Optional mask for applying attention.

        Returns:
            Tuple: Attention, instantaneous attention, and external variable attention tensors.
        """
        batch_size, _, seq_len = x.size()

        x = layer(x)

        if mask is None:
            x = x.reshape(batch_size, self.groups, self.n, seq_len)
        else:
            x = x.reshape(batch_size, self.n, self.n, seq_len).masked_fill(self.mask == 0, 0)

        attn = x[:, :self.n].permute(0, 2, 1, 3)
        attn_ext = x[:, self.n:].permute(0, 2, 1, 3)

        x = x.sum(dim=1)  # (batch_size, n, seq_len)

        return x, attn, attn_ext

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Performs a forward pass through the NavarInstant interpretation layer.

        Args:
            x (torch.Tensor): Input tensor of size (batch_size, in_channels, sequence_length).

        Returns:
            ModelOutput: Object containing the output tensors: 'x', 'attn', 'attn_instantaneous',
                         and 'attn_external_variables'.
        """
        batch_size, _, seq_len = x.size()
        x = self.pad(x)

        x0, attn, attn_ext = self.calc(x[..., :-1], self.contributions_t0)

        x = x.reshape(batch_size, self.groups, self.dim, seq_len + 1)[:, :self.n, :, 1:]
        x = x.reshape(batch_size, -1, seq_len)
        x1, attn1, _ = self.calc(x, self.contributions_t1, mask=self.mask)

        causal_matrix = attn.std(dim=-1).detach()
        causal_matrix += attn1.std(dim=-1).detach()
        causal_matrix = torch.cat((causal_matrix, attn_ext.std(dim=-1).detach()), dim=-1)

        return ModelOutput(
            x_pred=x0 + x1 + self.biases,
            attn=attn,
            attn_instantaneous=attn1,
            attn_external_variables=attn_ext,
            attn_loss=attn_regularization(attn, attn1),
            causal_matrix=causal_matrix
        )


class NavarVarInstant(InterpretModule):
    """
    Variation of Navar interpretation layer with both variational layer and instantaneous predictions.

    Args:
        in_channels (int): Number of input channels.
        groups (int): Number of groups in the model.
        num_external_variables (int): Number of external variables.
    """

    def __init__(self, in_channels: int, groups: int, num_external_variables: int):
        super().__init__(in_channels, groups, num_external_variables, use_variational_layer=True,
                         use_instantaneous_predictions=True)
        self.contributions_t0 = TemporalVariationalLayer(in_channels, groups * self.n, groups=groups)
        self.contributions_t1 = TemporalVariationalLayer(self.n * self.dim, self.n * self.n, groups=self.n)

        self.biases = nn.Parameter(torch.empty(1, self.n, 1))
        self.biases.data.fill_(0.0001)

        mask = 1 - torch.eye(self.n, self.n, dtype=torch.int).reshape(1, self.n, self.n, 1)
        self.register_buffer('mask', mask)

        self.pad = nn.ConstantPad1d((0, 1), 0)

    def calc(self, x: torch.Tensor, layer, biases=None, mask=None):
        """
        Calculates the attention, mean, standard deviation, and KL loss using a given layer.

        Args:
            x (torch.Tensor): Input tensor.
            layer: The layer to use for calculation.
            biases: Biases to add to the calculated attention and mean.
            mask: Optional mask for applying attention.

        Returns:
            Tuple: Attention, mean, standard deviation, KL loss, attention tensors.
        """
        batch_size, _, seq_len = x.size()

        x, mu, std, kl_loss = layer(x)

        if mask is None:
            x = x.reshape(batch_size, self.groups, self.n, seq_len)
            mu = mu.reshape(batch_size, self.groups, self.n, seq_len)
            std = std.reshape(batch_size, self.groups, self.n, seq_len)
        else:
            x = x.reshape(batch_size, self.n, self.n, seq_len).masked_fill(self.mask == 0, 0)
            mu = mu.reshape(batch_size, self.n, self.n, seq_len).masked_fill(self.mask == 0, 0)
            std = std.reshape(batch_size, self.n, self.n, seq_len).masked_fill(self.mask == 0, 0)

        attn = x[:, :self.n].permute(0, 2, 1, 3)
        attn_ext = x[:, self.n:].permute(0, 2, 1, 3)

        attn_mu = mu[:, :self.n].permute(0, 2, 1, 3)
        attn_ext_mu = mu[:, self.n:].permute(0, 2, 1, 3)

        mu, std = sum_of_distributions(mu, std, dim=1)  # (batch_size, n, seq_len)

        x = x.sum(dim=1)  # (batch_size, n, seq_len)

        if biases is not None:
            x += biases
            mu += biases

        return x, mu, std, kl_loss, attn, attn_ext, attn_mu, attn_ext_mu

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Performs a forward pass through the NavarVarInstant interpretation layer.

        Args:
            x (torch.Tensor): Input tensor of size (batch_size, in_channels, sequence_length).

        Returns:
            ModelOutput: Object containing the output tensors: 'x', 'mu', 'std', 'attn', 'attn_instantaneous',
                  'attn_external_variables', and 'loss'.
        """
        batch_size, _, seq_len = x.size()
        x = self.pad(x)

        x0, mu0, std0, kl_loss0, attn, attn_ext, attn_mu, attn_ext_mu = self.calc(x[..., :-1],
                                                                                  self.contributions_t0,
                                                                                  biases=self.biases)

        x = x.reshape(batch_size, self.groups, self.dim, seq_len + 1)[:, :self.n, :, 1:].reshape(batch_size, -1, seq_len)
        x1, mu1, std1, kl_loss1, attn1, _, attn1_mu, _ = self.calc(x,
                                                                   self.contributions_t1,
                                                                   mask=self.mask)

        causal_matrix = attn_mu.std(dim=-1).detach()
        causal_matrix += attn1_mu.std(dim=-1).detach()
        causal_matrix = torch.cat((causal_matrix, attn_ext_mu.std(dim=-1).detach()), dim=-1)

        return ModelOutput(
            x_pred=x0 + x1,
            mu_pred=mu0 + mu1,
            std_pred=torch.sqrt(std0.pow(2) + std1.pow(2)),
            attn=attn,
            attn_instantaneous=attn1,
            attn_external_variables=attn_ext,
            kl_loss=kl_loss0 + kl_loss1,
            attn_loss=attn_regularization(attn, attn1),
            causal_matrix=causal_matrix
        )
