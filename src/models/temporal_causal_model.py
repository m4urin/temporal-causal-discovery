import torch.nn as nn
import torch

from src.models.model_outputs import ModelOutput, ModelResult


class TemporalCausalModel(nn.Module):
    """
    A superclass for performing causal prediction in time series data.

    This model defines methods for getting the receptive field of the model, getting the number of parameters
    in the model, and evaluating the model on input data. To use this model, you need to implement the forward method.
    """
    def __init__(self, num_variables: int, receptive_field: int):
        super().__init__()
        self.num_variables = num_variables
        self.receptive_field = receptive_field

    @classmethod
    def get_hp_space(cls):
        raise NotImplementedError('Not implemented yet')

    def forward(self, x: torch.Tensor, max_batch_size: int = None) -> ModelOutput:
        """
        Computes the forward pass of the model.
        To use this model, you need to implement this method.
        """
        raise NotImplementedError('Not implemented yet')

    def get_result(self, x: torch.Tensor) -> ModelResult:
        raise NotImplementedError('Not implemented yet')

    def evaluate(self, x: torch.Tensor, max_batch_size: int = None) -> ModelOutput:
        with torch.no_grad():
            mode = self.training
            self.eval()
            model_output = self.forward(x, max_batch_size)
            self.train(mode)
            return model_output
