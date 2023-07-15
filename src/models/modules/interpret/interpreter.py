import torch
from torch import nn

from src.models.model_outputs import ModelOutput
from src.models.modules.interpret.attention_mechanism import AttentionDefault, AttentionInstant
from src.models.modules.interpret.navar import NavarVarInstant, NavarVar, NavarInstant, NavarDefault


class Interpreter(nn.Module):
    """
    Interpreter is a neural network module that interprets input data using different variants of Navar and
    Attention models. It provides flexibility in choosing between variational and default layers,
    as well as instantaneous predictions.

    Args:
        in_channels (int): Number of input channels.
        groups (int): Number of groups for grouping convolution.
        num_external_variables (int): Number of external variables.
        use_variational_layer (bool): Whether to use the variational layer.
        use_instantaneous_predictions (bool): Whether to use instantaneous predictions.

    Attributes:
        interpreter (nn.Module): Instance of the selected Navar model for interpretation.

    """
    def __init__(self, in_channels: int, groups: int, num_external_variables: int, use_navar: bool, use_attention: bool,
                 use_variational_layer: bool, use_instantaneous_predictions: bool, num_heads: int = None):
        super().__init__()

        if use_attention and use_navar:
            raise Exception('Can either use NAVAR contributions OR Attention mechanism, not both')
        elif use_navar:
            if use_variational_layer:
                if use_instantaneous_predictions:
                    self.interpreter = NavarVarInstant(in_channels, groups, num_external_variables)
                else:
                    self.interpreter = NavarVar(in_channels, groups, num_external_variables)
            else:
                if use_instantaneous_predictions:
                    self.interpreter = NavarInstant(in_channels, groups, num_external_variables)
                else:
                    self.interpreter = NavarDefault(in_channels, groups, num_external_variables)
        elif use_attention:
            if use_instantaneous_predictions:
                self.interpreter = AttentionInstant(
                    in_channels,
                    groups,
                    num_external_variables,
                    use_variational_layer,
                    num_heads
                )
            else:
                self.interpreter = AttentionDefault(
                    in_channels,
                    groups,
                    num_external_variables,
                    use_variational_layer,
                    num_heads
                )
        else:
            raise Exception('Must either use NAVAR contributions OR Attention mechanism')

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass of the Interpreter.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            ModelOutput: Output of the interpreter.

        """
        return self.interpreter(x)
