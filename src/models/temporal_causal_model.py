import torch.nn as nn
import torch
from src.utils.pytorch import count_parameters


class TemporalCausalModel(nn.Module):
    """
    A temporal causal model used as a superclass to perform causal prediction in time series data.

    Attributes:
    -----------
    None

    Methods:
    --------
    get_receptive_field(self) -> int:
        Returns the receptive field of the model.

    get_n_parameters(self, trainable_only: bool = False) -> int:
        Returns the number of parameters in the model.

        Parameters:
        -----------
        trainable_only : bool, optional
            If True, returns only the number of trainable parameters.

        Returns:
        --------
        int
            The number of parameters in the model.

    evaluate(self, x: torch.Tensor) -> dict:
        Evaluates the model on the input tensor x and returns a dictionary with evaluation data.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor.

        Returns:
        --------
        dict
            A dictionary with evaluation data: {'train_loss': float, 'test_loss': float, 'causal_matrix': {'variational_inference': float, 'monte_carlo': float}}
    """
    def __init__(self):
        super().__init__()

    def get_receptive_field(self) -> int:
        """
        Returns the receptive field of the model.

        The receptive field is defined as the number of input time steps that influence a single output time step.

        Parameters:
        -----------
        None

        Returns:
        --------
        int
            The receptive field of the model.
        """
        raise NotImplementedError('Not implemented yet')

    def get_n_parameters(self, trainable_only: bool = False) -> int:
        """
        Returns the number of parameters in the model.

        Parameters:
        -----------
        trainable_only : bool, optional
            If True, returns only the number of trainable parameters.

        Returns:
        --------
        int
            The number of parameters in the model.
        """
        return count_parameters(self, trainable_only)

    def evaluate(self, x: torch.Tensor) -> dict:
        """
        Evaluates the model on the input tensor x and returns a dictionary with evaluation data.

        The evaluation data includes the train and test losses, as well as the causal matrix computed using variational inference
        and Monte Carlo methods.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor.

        Returns:
        --------
        dict
            A dictionary with evaluation data: {
               'train_loss': float,
               'test_loss': float,
               'causal_matrix': {
                  'mu': list,
                  'variational_inference': list,
                  'monte_carlo': list}
            }
        """
        raise NotImplementedError('Not implemented yet')
