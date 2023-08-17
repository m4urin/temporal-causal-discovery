import torch
from torch import nn


from src.models.tcn import TCN
from src.utils.pytorch import count_parameters

# Neural Additive Vector Auto-Regression (NAVAR) is an additive model to identify causal relationships in time series data.
# This file contains various versions that includes uncertainty estimates:
# - NAVAR_Aleatoric learns the aleatoric uncertainty of the output predictions, with the goal of learning contributions that do not incorporate noise
# - NAVAR_Epistemic learns the epistemic uncertainty (based on the paper Deep Evidential Regression) in addition to the aleatoric uncertainty
# - Lastly, NAVAR_Uncertainty provides epistemic and aleatoric uncertainties to the contributions themselves and well as the final predictions. This might help in the construction of the causal matrix from the contributions.

class NAVAR(nn.Module):
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                 dropout=0.0, weight_sharing=False, recurrent=False,
                 aleatoric=False, epistemic=False, uncertainty_contributions=False, **kwargs):
        super().__init__()
        if uncertainty_contributions:
            self.navar = NAVAR_Uncertainty(n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                                           dropout, weight_sharing, recurrent)
        elif epistemic:
            self.navar = NAVAR_Epistemic(n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                                         dropout, weight_sharing, recurrent)
        elif aleatoric:
            self.navar = NAVAR_Aleatoric(n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                                         dropout, weight_sharing, recurrent)
        else:
            self.navar = NAVAR_Default(n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                                       dropout, weight_sharing, recurrent)

        self.receptive_field = self.navar.receptive_field
        self.n_params = count_parameters(self)

    def forward(self, x_input):
        return self.navar(x_input)

    def loss_function(self, *args, **kwargs):
        return self.navar.loss_function(*args, **kwargs)

    def analysis(self, x_input):
        with torch.no_grad():
            return self.navar.analysis(x_input)


class NAVAR_Default(nn.Module):
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                 dropout=0.0, weight_sharing=False, recurrent=False):
        super().__init__()
        self.contributions = TCN(in_channels=n_variables, out_channels=n_variables * n_variables,
                                 hidden_dim=n_variables * hidden_dim, kernel_size=kernel_size,
                                 n_blocks=n_blocks, n_layers_per_block=n_layers_per_block, groups=n_variables,
                                 dropout=dropout, weight_sharing=weight_sharing, recurrent=recurrent)
        self.biases = nn.Parameter(torch.ones(1, n_variables, 1) * 0.01)

    def forward(self, x_input):
        batch_size, n, seq = x_input.size()

        # (batch_size, n, seq) -> (batch_size, n * n, seq) -> (batch_size, n, n, seq)
        contributions = self.contributions(x_input).reshape(batch_size, n, n, seq)
        prediction = contributions.sum(dim=1) + self.biases

        return prediction, contributions

    def loss_function(self, y_true, prediction, contributions, lambda1=0.2):
        error = nn.functional.mse_loss(prediction, y_true)
        regularization = contributions.abs().mean()
        return error + lambda1 * regularization

    def analysis(self, x_input):
        prediction, contributions = self.forward(x_input)
        return prediction, contributions


class NAVAR_Aleatoric(nn.Module):
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                 dropout=0.0, weight_sharing=False, recurrent=False):
        super().__init__()

        self.contributions = TCN(in_channels=n_variables, out_channels=n_variables * n_variables,
                                 hidden_dim=n_variables * hidden_dim, kernel_size=kernel_size,
                                 n_blocks=n_blocks, n_layers_per_block=n_layers_per_block, groups=n_variables,
                                 dropout=dropout, weight_sharing=weight_sharing, recurrent=recurrent)
        self.aleatoric = TCN(in_channels=n_variables, out_channels=n_variables,
                             hidden_dim=2 * hidden_dim, kernel_size=kernel_size,
                             n_blocks=n_blocks, n_layers_per_block=n_layers_per_block, groups=1,
                             dropout=dropout, weight_sharing=weight_sharing, recurrent=recurrent)
        self.biases = nn.Parameter(torch.ones(1, n_variables, 1) * 0.01)

    def forward(self, x_input):
        batch_size, n, seq = x_input.size()

        log_var_aleatoric = self.aleatoric(x_input)  # (batch_size, n, seq)

        # (batch_size, n, seq) -> (batch_size, n * n, seq) -> (batch_size, n, n, seq)
        contributions = self.contributions(x_input).reshape(batch_size, n, n, seq)
        prediction = contributions.sum(dim=1) + self.biases

        return prediction, contributions, log_var_aleatoric

    def loss_function(self, y_true, prediction, contributions, log_var_aleatoric, lambda1=0.2):
        aleatoric_loss = torch.mean(log_var_aleatoric + (prediction - y_true).pow(2) * torch.exp(-log_var_aleatoric))
        regularization = contributions.abs().mean()
        return aleatoric_loss + lambda1 * regularization

    def analysis(self, x_input):
        prediction, contributions, log_var_aleatoric = self.forward(x_input)
        aleatoric = torch.exp(0.5 * log_var_aleatoric)

        return prediction, contributions, aleatoric


class NAVAR_Epistemic(nn.Module):
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                 dropout=0.0, weight_sharing=False, recurrent=False):
        super().__init__()

        self.contributions = TCN(in_channels=n_variables, out_channels=n_variables * n_variables,
                                 hidden_dim=n_variables * hidden_dim, kernel_size=kernel_size,
                                 n_blocks=n_blocks, n_layers_per_block=n_layers_per_block, groups=n_variables,
                                 dropout=dropout, weight_sharing=weight_sharing, recurrent=recurrent)
        self.epistemic = TCN(in_channels=n_variables, out_channels=2 * n_variables,
                             hidden_dim=2 * hidden_dim, kernel_size=kernel_size,
                             n_blocks=n_blocks, n_layers_per_block=n_layers_per_block, groups=1,
                             dropout=dropout, weight_sharing=weight_sharing, recurrent=recurrent)
        self.biases = nn.Parameter(torch.ones(1, n_variables, 1) * 0.01)

    def forward(self, x_input):
        batch_size, n, seq = x_input.size()

        log_v, log_beta = self.aleatoric(x_input).chunk(chunks=2, dim=1)  # (batch_size, n, seq)
        log_var_aleatoric = log_beta - log_v

        # (batch_size, n, seq) -> (batch_size, n * n, seq) -> (batch_size, n, n, seq)
        contributions = self.contributions(x_input).reshape(batch_size, n, n, seq)
        prediction = contributions.sum(dim=1) + self.biases

        return prediction, contributions, log_var_aleatoric, log_v

    def loss_function(self, y_true, prediction, contributions, log_var_aleatoric, log_v, lambda1=0.2, coeff=1e-1):
        error = (1.0 + coeff * torch.exp(log_v)) * (prediction - y_true).pow(2)
        epistemic_loss = torch.mean(log_var_aleatoric + error * torch.exp(-log_var_aleatoric))
        regularization = contributions.abs().mean()
        return epistemic_loss + lambda1 * regularization

    def analysis(self, x_input):
        prediction, contributions, log_var_aleatoric, log_v = self.forward(x_input)
        aleatoric = torch.exp(0.5 * log_var_aleatoric)
        epistemic = torch.exp(-0.5 * log_v)

        return prediction, contributions, aleatoric, epistemic


class NAVAR_Uncertainty(nn.Module):
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                 dropout=0.0, weight_sharing=False, recurrent=False):
        super().__init__()

        self.contributions = TCN(in_channels=n_variables, out_channels=3 * n_variables * n_variables,
                                 hidden_dim=n_variables * hidden_dim, kernel_size=kernel_size,
                                 n_blocks=n_blocks, n_layers_per_block=n_layers_per_block, groups=n_variables,
                                 dropout=dropout, weight_sharing=weight_sharing, recurrent=recurrent)
        self.uncertainty = TCN(in_channels=n_variables, out_channels=2 * n_variables,
                               hidden_dim=2 * hidden_dim, kernel_size=kernel_size,
                               n_blocks=n_blocks, n_layers_per_block=n_layers_per_block, groups=1,
                               dropout=dropout, weight_sharing=weight_sharing, recurrent=recurrent)
        self.biases = nn.Parameter(torch.ones(1, n_variables, 1) * 0.01)

    def forward(self, x_input):
        batch_size, n, seq = x_input.size()

        # (batch_size, n, seq) -> (batch_size, n * n, seq) -> (batch_size, n, n, seq)
        contributions, log_v_contr, log_beta_contr = self.contributions(x_input).reshape(batch_size, n, 3 * n, seq).chunk(chunks=3, dim=2)
        log_var_aleatoric_contr = log_beta_contr - log_v_contr

        log_v, log_beta = self.uncertainty(x_input).reshape(batch_size, n, 2 * n, seq).chunk(chunks=2, dim=2)
        log_var_aleatoric = log_beta - log_v

        prediction = contributions.sum(dim=1) + self.biases

        return prediction, contributions, log_var_aleatoric, log_v, log_var_aleatoric_contr, log_v_contr

    def loss_function(self, y_true, prediction, contributions, log_var_aleatoric, log_v,
                      log_var_aleatoric_contr, log_v_contr, lambda1=0.2, coeff=1e-1):
        error = (1.0 + coeff * torch.exp(log_v_contr)) * (contributions - y_true.unsqueeze(1)).pow(2)
        epistemic_loss_contr = torch.mean(log_var_aleatoric_contr + error * torch.exp(-log_var_aleatoric_contr))

        error_out = (1.0 + coeff * torch.exp(log_v)) * (prediction - y_true).pow(2)
        epistemic_loss = torch.mean(log_var_aleatoric + error_out * torch.exp(-log_var_aleatoric))

        regularization = contributions.abs().mean()

        return epistemic_loss_contr + epistemic_loss + lambda1 * regularization

    def analysis(self, x_input):
        prediction, contributions, log_var_aleatoric, log_v, log_var_aleatoric_contr, log_v_contr = self.forward(x_input)
        aleatoric = torch.exp(0.5 * log_var_aleatoric)
        aleatoric_contr = torch.exp(0.5 * log_var_aleatoric_contr)
        epistemic = torch.exp(-0.5 * log_v)
        epistemic_contr = torch.exp(-0.5 * log_v_contr)

        return prediction, contributions, aleatoric, epistemic, aleatoric_contr, epistemic_contr
