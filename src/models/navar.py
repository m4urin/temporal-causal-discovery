import torch
from torch import nn
from src.models.TCN import TCN
from src.losses import DER_loss, NAVAR_regularization_loss, NLL_loss
from src.utils import weighted_std, sliding_window_std, weighted_sliding_window_std, count_parameters


class NAVAR(nn.Module):
    """
    Neural Additive Vector Auto-Regression (NAVAR) model wrapper.

    This class acts as a wrapper for different variations of the NAVAR model,
    each handling different types of uncertainties and information.
    - NAVAR_Aleatoric: learns aleatoric uncertainty of output predictions
    - NAVAR_Epistemic: learns epistemic uncertainty in addition to aleatoric uncertainty of output predictions
    - NAVAR_Uncertainty: provides epistemic and eleatoric uncertainties over the contributions and final predictions

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
        loss_function(y_true, **kwargs): Compute the loss for training.
        analysis(**kwargs): Perform analysis using the NAVAR model.
    """
    def __init__(self, n_variables, hidden_dim, kernel_size, n_blocks, n_layers_per_block,
                 dropout=0.0, weight_sharing=False, recurrent=False,
                 aleatoric=False, epistemic=False, uncertainty_contributions=False, n_heads=None, softmax_method=None):
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

        self.receptive_field = (2 ** n_blocks - 1) * n_layers_per_block * (kernel_size - 1) + 1
        self.n_params = count_parameters(self)

    def forward(self, x):
        return self.navar(x)

    def loss_function(self, y_true, coeff, beta, **kwargs):
        return self.navar.loss_function(y_true, **kwargs)

    def analysis(self, **kwargs):
        mode = self.training
        self.eval()
        with torch.no_grad():
            result = self.navar.analysis(**kwargs)
            #result = {k: v.cpu().numpy() for k, v in result.items()}
        self.train(mode)
        return result


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

    def forward(self, x):
        batch_size, n, seq = x.size()

        # Calculate contributions: (batch_size, n, seq) -> (batch_size, n * n, seq) -> (batch_size, n, n, seq)
        contributions = self.contributions(x).reshape(batch_size, n, n, seq)

        # Sum contributions and add biases
        prediction = contributions.sum(dim=1) + self.biases

        return {
            'prediction': prediction,
            'contributions': contributions.transpose(1, 2)
        }

    @staticmethod
    def analysis(prediction, contributions):
        # Calculate aleatoric and epistemic uncertainties
        # contributions: (bs, n, n, sequence_length)
        temp_causal_matrix = sliding_window_std(contributions, window=(25, 25), dim=-1)  # (bs, n, n, sequence_length)
        default_causal_matrix = contributions.std(dim=-1)  # (bs, n, n)

        return {
            'prediction': prediction,
            'contributions': contributions,
            'default_causal_matrix': default_causal_matrix,
            'temp_causal_matrix': temp_causal_matrix
        }

    @staticmethod
    def loss_function(y_true, prediction, contributions, lambda1=0.2):
        # Mean squared error loss
        loss = nn.functional.mse_loss(prediction, y_true)
        regularization = NAVAR_regularization_loss(contributions, lambda1=lambda1)

        return loss + regularization


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

    def forward(self, x):
        batch_size, n, seq = x.size()

        # Calculate aleatoric uncertainty
        log_var_aleatoric = self.aleatoric(x)  # (batch_size, n, seq)

        # Calculate contributions
        contributions = self.contributions(x).reshape(batch_size, n, n, seq)

        # Sum contributions and add biases
        prediction = contributions.sum(dim=1) + self.biases

        return {
            'prediction': prediction,
            'contributions': contributions.transpose(1, 2),
            'log_var_aleatoric': log_var_aleatoric
        }

    @staticmethod
    def analysis(prediction, contributions, log_var_aleatoric):
        # Calculate aleatoric and epistemic uncertainties
        # contributions: (bs, n, n, sequence_length)
        aleatoric = torch.exp(0.5 * log_var_aleatoric)  # (bs, n, sequence_length))

        temp_causal_matrix = sliding_window_std(contributions, window=(25, 25), dim=-1)  # (bs, n, n, sequence_length)
        default_causal_matrix = contributions.std(dim=-1)  # (bs, n, n)

        return {
            'prediction': prediction,
            'contributions': contributions,
            'aleatoric': aleatoric,
            'default_causal_matrix': default_causal_matrix,
            'temp_causal_matrix': temp_causal_matrix
        }

    @staticmethod
    def loss_function(y_true, prediction, contributions, log_var_aleatoric, lambda1=0.2):
        aleatoric_loss = NLL_loss(y_true=y_true, y_pred=prediction, log_var=log_var_aleatoric)
        regularization = NAVAR_regularization_loss(contributions, lambda1=lambda1)

        return aleatoric_loss + regularization


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

    def forward(self, x):
        batch_size, n, seq = x.size()

        log_v, log_beta = self.epistemic(x).chunk(2, dim=2)  # (batch_size, n, seq)
        log_var_aleatoric = log_beta - log_v

        # Calculate contributions
        contributions = self.contributions(x).reshape(batch_size, n, n, seq)

        # Sum contributions and add biases
        prediction = contributions.sum(dim=1) + self.biases

        return {
            'prediction': prediction,
            'contributions': contributions.transpose(1, 2),
            'log_var_aleatoric': log_var_aleatoric,
            'log_v': log_v
        }

    @staticmethod
    def analysis(prediction, contributions, log_var_aleatoric, log_v):
        # Calculate aleatoric and epistemic uncertainties
        # contributions: (bs, n, n, sequence_length)
        aleatoric = torch.exp(0.5 * log_var_aleatoric)  # (bs, n, sequence_length)
        epistemic = torch.exp(-0.5 * log_v)  # (bs, n, sequence_length)
        epistemic_contr = epistemic.unsqueeze(2).expand(-1, -1, epistemic.size(1), -1)  # (bs, n, n, sequence_length)

        temp_confidence_matrix = 1 / epistemic_contr  # (bs, n, n, sequence_length)
        temp_causal_matrix = weighted_sliding_window_std(contributions, weights=temp_confidence_matrix,
                                                         window=(25, 25), dim=-1)  # (bs, n, n, sequence_length)

        default_confidence_matrix = temp_confidence_matrix.mean(dim=-1)  # (bs, n, n)
        default_causal_matrix = weighted_std(contributions, weights=temp_confidence_matrix, dim=-1)  # (bs, n, n)

        return {
            'prediction': prediction,
            'contributions': contributions,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'default_causal_matrix': default_causal_matrix,
            'default_confidence_matrix': default_confidence_matrix,
            'temp_causal_matrix': temp_causal_matrix,
            'temp_confidence_matrix': temp_confidence_matrix,
        }

    @staticmethod
    def loss_function(y_true, prediction, contributions, log_var_aleatoric, log_v, lambda1=0.2, coeff=1e-1):
        # Calculate epistemic uncertainty term
        epistemic_loss = DER_loss(y_true=y_true, y_pred=prediction, log_var=log_var_aleatoric, log_v=log_v, coeff=coeff)
        regularization = NAVAR_regularization_loss(contributions, lambda1=lambda1)

        return epistemic_loss + regularization


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

    def forward(self, x):
        batch_size, n, seq = x.size()

        # Calculate uncertainties for individual contributions
        contr = self.contributions(x).reshape(batch_size, n, 3 * n, seq)
        contributions, log_v_contr, log_beta_contr = contr.chunk(chunks=3, dim=2)

        log_var_aleatoric_contr = log_beta_contr - log_v_contr

        log_v, log_beta = self.uncertainty(x).reshape(batch_size, n, 2 * n, seq).chunk(chunks=2, dim=2)
        log_var_aleatoric = log_beta - log_v

        # Calculate contributions and predictions
        prediction = contributions.sum(dim=1) + self.biases

        return {
            'prediction': prediction,
            'contributions': contributions.transpose(1, 2),
            'log_var_aleatoric': log_var_aleatoric,
            'log_v': log_v,
            'log_var_aleatoric_contr': log_var_aleatoric_contr.transpose(1, 2),
            'log_v_contr': log_v_contr.transpose(1, 2)
        }

    @staticmethod
    def analysis(prediction, contributions, log_var_aleatoric, log_v, log_var_aleatoric_contr, log_v_contr):
        # Calculate aleatoric and epistemic uncertainties
        # contributions: (bs, n, n, sequence_length)
        aleatoric_contr = torch.exp(0.5 * log_var_aleatoric_contr)  # (bs, n, n, sequence_length)
        epistemic_contr = torch.exp(-0.5 * log_v_contr)  # (bs, n, n, sequence_length)
        aleatoric = torch.exp(0.5 * log_var_aleatoric)  # (bs, n, sequence_length)
        epistemic = torch.exp(-0.5 * log_v)  # (bs, n, sequence_length)

        temp_confidence_matrix = 1 / epistemic_contr  # (bs, n, n, sequence_length)

        a1 = contributions + 2 * aleatoric_contr
        a2 = contributions - 2 * aleatoric_contr

        temp_c1 = weighted_sliding_window_std(a1, weights=temp_confidence_matrix, window=(25, 25), dim=-1)
        temp_c2 = weighted_sliding_window_std(a2, weights=temp_confidence_matrix, window=(25, 25), dim=-1)
        temp_causal_matrix = (temp_c1 + temp_c2) / 2  # (bs, n, n, sequence_length)

        default_confidence_matrix = temp_confidence_matrix.mean(dim=-1)  # (bs, n, n)

        default_c1 = weighted_std(a1, weights=temp_confidence_matrix, dim=-1)
        default_c2 = weighted_std(a2, weights=temp_confidence_matrix, dim=-1)
        default_causal_matrix = (default_c1 + default_c2) / 2  # (bs, n, n, sequence_length)

        return {
            'prediction': prediction,
            'contributions': contributions,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'aleatoric_contr': aleatoric_contr,
            'epistemic_contr': epistemic_contr,
            'default_causal_matrix': default_causal_matrix,
            'default_confidence_matrix': default_confidence_matrix,
            'temp_causal_matrix': temp_causal_matrix,
            'temp_confidence_matrix': temp_confidence_matrix,
        }

    @staticmethod
    def loss_function(y_true, prediction, contributions, log_var_aleatoric, log_v,
                      log_var_aleatoric_contr, log_v_contr, lambda1=0.2, coeff=1e-1):
        # Calculate epistemic uncertainty term for individual contributions
        epistemic_loss_contr = DER_loss(y_true=y_true.unsqueeze(1), y_pred=contributions,
                                        log_var=log_var_aleatoric_contr, log_v=log_v_contr, coeff=coeff)
        epistemic_loss = DER_loss(y_true=y_true, y_pred=prediction, log_var=log_var_aleatoric, log_v=log_v, coeff=coeff)
        regularization = NAVAR_regularization_loss(contributions, lambda1=lambda1)

        return epistemic_loss_contr + epistemic_loss + regularization
