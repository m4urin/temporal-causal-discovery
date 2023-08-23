import torch
import torch.nn as nn

from src.models.config.model_outputs import ModelOutput


class InterpretModule(nn.Module):
    """
    Final layer of the model that handles the interpretation of the predictions to generate the causal matrix.

    Args:
        in_channels (int): Number of input channels.
        groups (int): Number of groups in the model.
        num_external_variables (int): Number of external variables.
        use_variational_layer (bool, optional): Whether to use a variational layer. Defaults to False.
        use_instantaneous_predictions (bool, optional): Whether to use instantaneous predictions. Defaults to False.
    """
    def __init__(
        self,
        in_channels: int,
        groups: int,
        num_external_variables: int,
        use_variational_layer: bool = False,
        use_instantaneous_predictions: bool = False,
    ):
        super().__init__()
        self.use_variational_layer = use_variational_layer
        self.use_instantaneous_predictions = use_instantaneous_predictions
        self.groups = groups
        self.n = groups - num_external_variables
        self.e = num_external_variables
        self.dim = in_channels // groups

    def forward(self, x: torch.Tensor) -> ModelOutput:
        raise NotImplementedError('Not implemented yet.')
