import torch
from torch import nn

from src.models.TAMCaD import TAMCaD
from src.models.NAVAR import NAVAR
from src.utils import count_parameters, receptive_field


class CausalModel(nn.Module):
    """
    A wrapper class for causal models like NAVAR and TAMCaD.

    Args:
        model (str): The type of causal model to be used ('NAVAR', 'TAMCaD', 'Simple').
        architecture (dict): Dictionary containing architecture parameters like
                             'kernel_size', 'n_layers_per_block', and 'n_blocks'.
        lambda1 (float): Regularization hyper-parameter.

    Keyword Args:
        n_variables (int): Number of variables in the time series.
        hidden_dim (int): Hidden dimension size for the submodels.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        weight_sharing (bool, optional): Whether to use weight sharing in TCN. Defaults to False.
        recurrent (bool, optional): Whether to use recurrent layers in the TCN. Defaults to False.
        lambda1 (float): Regularization hyper-parameter.

    Attributes:
        causal_model (nn.Module): Instance of the specific causal model specified by 'model'.
        n_params (int): Number of parameters in the model.
        receptive_field (int): The size of the receptive field for the model.

    Methods:
        forward(x): Forward pass through the specified causal model.
        loss_function(y_true, **kwargs): Compute the loss for training.
        analysis(**kwargs): Perform analysis using the specified causal model.
    """
    def __init__(self, model, architecture, **kwargs):
        super().__init__()

        # Initialize the causal model based on the 'model' argument
        if model == 'NAVAR':
            self.causal_model = NAVAR(**architecture, **kwargs)
        elif model == 'TAMCaD':
            self.causal_model = TAMCaD(**architecture, **kwargs)
        elif model == 'Simple':
            self.causal_model = None  # TODO
        else:
            raise ValueError("Invalid model type specified.")

        self.n_params = count_parameters(self)  # Count number of parameters
        self.receptive_field = receptive_field(**architecture)  # Calculate receptive field size

    def forward(self, x):
        return self.causal_model(x)

    def loss_function(self, y_true, **kwargs):
        return self.causal_model.loss_function(y_true, **kwargs)

    def analysis(self, x):
        # Store current training mode and set model to eval mode for analysis
        mode = self.training
        self.eval()

        # Perform analysis without gradient computation
        with torch.no_grad():
            result = self.causal_model.analysis(**self.causal_model(x))

        # Revert to original mode after analysis
        self.train(mode)

        return result
