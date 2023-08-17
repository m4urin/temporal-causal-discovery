import torch
from torch import nn
from src.models.tcn import TCN
from src.utils.pytorch import count_parameters


class NAVAR(nn.Module):
    """
    Neural Additive Vector Auto-Regression (NAVAR) model wrapper.

    This class acts as a wrapper for different variations of the NAVAR model,
    each handling different types of uncertainties and information.
    - NAVAR_Aleatoric: learns aleatoric uncertainty of output predictions
    - NAVAR_Epistemic: learns epistemic uncertainty in addition to aleatoric uncertainty of output predictions
    - NAVAR_Uncertainty: provides epistemic and eleatoric uncertainties of individual contributions and final predictions

    Args:
        n_variables (int): Number of variables in the time series.
        hidden_dim (int): Hidden dimension size for the submodels.
        kernel_size (int): Size of the convolutional kernel.
        n_blocks (int): Number of blocks in the TCN architecture.
        n_layers_per_block (int): Number of layers per block in the TCN architecture.
        dropout (float, optional): Dropout probability. Default is 0.0.
        weight_sharing (bool, optional): Whether to use weight sharing in TCN. Default is False.
        recurrent (bool, optional): Whether to use recurrent layers in the TCN. Default is False.
        aleatoric (bool, optional): Whether to use aleatoric uncertainty. Default is False.
        epistemic (bool, optional): Whether to use epistemic uncertainty. Default is False.
        uncertainty_contributions (bool, optional): Whether to provide uncertainty for contributions. Default is False.

    Methods:
        forward(x_input): Forward pass through the NAVAR model.
        loss_function(*args, **kwargs): Compute the loss for training.
        analysis(x_input): Perform analysis using the NAVAR model.
    """

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

        # Retrieve attributes from the selected NAVAR variation
        self.receptive_field = self.navar.receptive_field
        self.n_params = count_parameters(self)

    def forward(self, x_input):
        """
        Forward pass through the NAVAR model.

        Args:
            x_input (torch.Tensor): Input time series data of shape (batch_size, n_variables, sequence_length).

        Returns:
            tuple: Tuple containing the results.
        """
        return self.navar(x_input)

    def loss_function(self, y_true, *args, **kwargs):
        return self.navar.loss_function(y_true, *args, **kwargs)

    def analysis(self, x_input):
        with torch.no_grad():
            return self.navar.analysis(x_input)


class NAVAR_Default(nn.Module):
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                 dropout=0.0, weight_sharing=False, recurrent=False):
        super().__init__()

        # TCN to learn additive contributions
        self.contributions = TCN(
            in_channels=n_variables,
            out_channels=n_variables * n_variables,
            hidden_dim=n_variables * hidden_dim,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            groups=n_variables,
            dropout=dropout,
            weight_sharing=weight_sharing,
            recurrent=recurrent
        )

        # Learnable biases
        self.biases = nn.Parameter(torch.ones(1, n_variables, 1) * 0.01)

    def forward(self, x_input):
        batch_size, n, seq = x_input.size()

        # Calculate contributions: (batch_size, n, seq) -> (batch_size, n * n, seq) -> (batch_size, n, n, seq)
        contributions = self.contributions(x_input).reshape(batch_size, n, n, seq)

        # Sum contributions and add biases
        prediction = contributions.sum(dim=1) + self.biases

        return prediction, contributions

    def loss_function(self, y_true, prediction, contributions, lambda1=0.2):
        # Mean squared error loss
        error = nn.functional.mse_loss(prediction, y_true)

        # Regularization term based on absolute contributions
        regularization = contributions.abs().mean()

        return error + lambda1 * regularization

    def analysis(self, x_input):
        prediction, contributions = self.forward(x_input)
        return prediction, contributions


class NAVAR_Aleatoric(nn.Module):
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                 dropout=0.0, weight_sharing=False, recurrent=False):
        super().__init__()

        # TCN to learn additive contributions
        self.contributions = TCN(
            in_channels=n_variables,
            out_channels=n_variables * n_variables,
            hidden_dim=n_variables * hidden_dim,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            groups=n_variables,
            dropout=dropout,
            weight_sharing=weight_sharing,
            recurrent=recurrent
        )

        # TCN to learn aleatoric uncertainty
        self.aleatoric = TCN(
            in_channels=n_variables,
            out_channels=n_variables,
            hidden_dim=2 * hidden_dim,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            groups=1,
            dropout=dropout,
            weight_sharing=weight_sharing,
            recurrent=recurrent
        )

        # Learnable biases
        self.biases = nn.Parameter(torch.ones(1, n_variables, 1) * 0.01)

    def forward(self, x_input):
        batch_size, n, seq = x_input.size()

        # Calculate aleatoric uncertainty
        log_var_aleatoric = self.aleatoric(x_input)  # (batch_size, n, seq)

        # Calculate contributions
        contributions = self.contributions(x_input).reshape(batch_size, n, n, seq)

        # Sum contributions and add biases
        prediction = contributions.sum(dim=1) + self.biases

        return prediction, contributions, log_var_aleatoric

    def loss_function(self, y_true, prediction, contributions, log_var_aleatoric, lambda1=0.2):
        # Calculate aleatoric loss
        aleatoric_loss = torch.mean(log_var_aleatoric + (prediction - y_true).pow(2) * torch.exp(-log_var_aleatoric))

        # Regularization term based on absolute contributions
        regularization = contributions.abs().mean()

        return aleatoric_loss + lambda1 * regularization

    def analysis(self, x_input):
        prediction, contributions, log_var_aleatoric = self.forward(x_input)

        # Calculate aleatoric uncertainty
        aleatoric = torch.exp(0.5 * log_var_aleatoric)

        return prediction, contributions, aleatoric


class NAVAR_Epistemic(nn.Module):
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                 dropout=0.0, weight_sharing=False, recurrent=False):
        super().__init__()

        # TCN to learn additive contributions
        self.contributions = TCN(
            in_channels=n_variables,
            out_channels=n_variables * n_variables,
            hidden_dim=n_variables * hidden_dim,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            groups=n_variables,
            dropout=dropout,
            weight_sharing=weight_sharing,
            recurrent=recurrent
        )

        # TCN to learn epistemic uncertainty
        self.epistemic = TCN(
            in_channels=n_variables,
            out_channels=2 * n_variables,
            hidden_dim=2 * hidden_dim,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            groups=1,
            dropout=dropout,
            weight_sharing=weight_sharing,
            recurrent=recurrent
        )

        # Learnable biases
        self.biases = nn.Parameter(torch.ones(1, n_variables, 1) * 0.01)

    def forward(self, x_input):
        batch_size, n, seq = x_input.size()

        log_v, log_beta = self.epistemic(x_input).chunk(chunks=2, dim=1)  # (batch_size, n, seq)
        log_var_aleatoric = log_beta - log_v

        # Calculate contributions
        contributions = self.contributions(x_input).reshape(batch_size, n, n, seq)

        # Sum contributions and add biases
        prediction = contributions.sum(dim=1) + self.biases

        return prediction, contributions, log_var_aleatoric, log_v

    def loss_function(self, y_true, prediction, contributions, log_var_aleatoric, log_v, lambda1=0.2, coeff=1e-1):
        # Calculate epistemic uncertainty term
        error = (1.0 + coeff * torch.exp(log_v)) * (prediction - y_true).pow(2)
        epistemic_loss = torch.mean(log_var_aleatoric + error * torch.exp(-log_var_aleatoric))

        # Regularization term based on absolute contributions
        regularization = contributions.abs().mean()

        return epistemic_loss + lambda1 * regularization

    def analysis(self, x_input):
        prediction, contributions, log_var_aleatoric, log_v = self.forward(x_input)

        # Calculate aleatoric and epistemic uncertainties
        aleatoric = torch.exp(0.5 * log_var_aleatoric)
        epistemic = torch.exp(-0.5 * log_v)

        return prediction, contributions, aleatoric, epistemic


class NAVAR_Uncertainty(nn.Module):
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                 dropout=0.0, weight_sharing=False, recurrent=False):
        super().__init__()

        # TCN to learn individual additive contributions
        self.contributions = TCN(
            in_channels=n_variables,
            out_channels=3 * n_variables * n_variables,
            hidden_dim=n_variables * hidden_dim,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            groups=n_variables,
            dropout=dropout,
            weight_sharing=weight_sharing,
            recurrent=recurrent
        )

        # TCN to learn individual uncertainties
        self.uncertainty = TCN(
            in_channels=n_variables,
            out_channels=2 * n_variables,
            hidden_dim=2 * hidden_dim,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            n_layers_per_block=n_layers_per_block,
            groups=1,
            dropout=dropout,
            weight_sharing=weight_sharing,
            recurrent=recurrent
        )

        # Learnable biases
        self.biases = nn.Parameter(torch.ones(1, n_variables, 1) * 0.01)

    def forward(self, x_input):
        batch_size, n, seq = x_input.size()

        # Calculate uncertainties for individual contributions
        contributions, log_v_contr, log_beta_contr = self.contributions(x_input).reshape(batch_size, n, 3 * n, seq).chunk(chunks=3, dim=2)
        log_var_aleatoric_contr = log_beta_contr - log_v_contr

        log_v, log_beta = self.uncertainty(x_input).reshape(batch_size, n, 2 * n, seq).chunk(chunks=2, dim=2)
        log_var_aleatoric = log_beta - log_v

        # Calculate contributions and predictions
        prediction = contributions.sum(dim=1) + self.biases

        return prediction, contributions, log_var_aleatoric, log_v, log_var_aleatoric_contr, log_v_contr

    def loss_function(self, y_true, prediction, contributions, log_var_aleatoric, log_v,
                      log_var_aleatoric_contr, log_v_contr, lambda1=0.2, coeff=1e-1):
        # Calculate epistemic uncertainty term for individual contributions
        error = (1.0 + coeff * torch.exp(log_v_contr)) * (contributions - y_true.unsqueeze(1)).pow(2)
        epistemic_loss_contr = torch.mean(log_var_aleatoric_contr + error * torch.exp(-log_var_aleatoric_contr))

        # Calculate epistemic uncertainty term for predictions
        error_out = (1.0 + coeff * torch.exp(log_v)) * (prediction - y_true).pow(2)
        epistemic_loss = torch.mean(log_var_aleatoric + error_out * torch.exp(-log_var_aleatoric))

        # Regularization term based on absolute contributions
        regularization = contributions.abs().mean()

        return epistemic_loss_contr + epistemic_loss + lambda1 * regularization

    def analysis(self, x_input):
        prediction, contributions, log_var_aleatoric, log_v, log_var_aleatoric_contr, log_v_contr = self.forward(x_input)

        # Calculate aleatoric and epistemic uncertainties
        aleatoric = torch.exp(0.5 * log_var_aleatoric)
        aleatoric_contr = torch.exp(0.5 * log_var_aleatoric_contr)
        epistemic = torch.exp(-0.5 * log_v)
        epistemic_contr = torch.exp(-0.5 * log_v_contr)

        return prediction, contributions, aleatoric, epistemic, aleatoric_contr, epistemic_contr
