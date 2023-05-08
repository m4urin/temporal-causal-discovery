import torch
from typing import Optional, List

from torch import nn

from src.data.dataset import Dataset
from src.models.model_config import ModelConfig, TrainConfig
from src.utils.pytorch import count_parameters


class ModelOutput:
    def __init__(self, predictions: torch.Tensor, losses: torch.Tensor):
        """
        Class to store model output. This class only contains tensors.

        Args:
            predictions : torch.Tensor
                Model output.
                Shape: (batch_size, num_var, sequence_length)
            losses : torch.Tensor
                Model loss.
                Shape: (batch_size,)
        """
        self.predictions: torch.Tensor = predictions
        self.losses: torch.Tensor = losses

    def get_loss(self, keep_batch=False) -> torch.Tensor:
        """
        Returns the model loss.

        Args:
            keep_batch : bool
                Whether to keep the batch dimension. Default is False.

        Returns:
            torch.Tensor
                Model loss.
                Shape: () if keep_batch is False, or (batch_size,) otherwise.
        """
        if keep_batch:
            return self.losses
        return self.losses.sum()  # sum all losses from various batches

    def items(self):
        """
        Yields the attribute names and their corresponding tensor values.

        Yields:
            Tuple
                The attribute name and its corresponding tensor value.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                yield attr_name, attr

    def cpu(self):
        """
        Moves all tensors to the CPU.
        """
        for name, t in self.items():
            setattr(self, name, t.detach().cpu())
        return self

    def cuda(self):
        """
        Moves all tensors to the GPU.
        """
        for name, t in self.items():
            setattr(self, name, t.cuda())
        return self


class ModelResult:
    """
    The final result of a model, including the time series prediction, the causal matrix found, and possibly the
    uncertainty of the predictions made.

    Parameters:
    -----------
    uncertainty_method : str
        Method used to estimate uncertainty.
        Must be one of 'eval', 'monte_carlo', or 'variational_inference'.

    true_data : torch.Tensor
        Ground truth data used for experiments and testing the model.

    predicted_data : torch.Tensor
        Predicted data generated by the model.

    causal_matrix : torch.Tensor
        Causal matrix found by the model.

    train_loss : List[float]
        Training loss during experiments.

    test_loss : List[float]
        Test loss during experiments.

    predicted_data_std : Optional[torch.Tensor]
        Standard deviation of predicted data. Defaults to None.

    causal_matrix_std : Optional[torch.Tensor]
        Standard deviation of causal matrix. Defaults to None.
    """

    def __init__(self, uncertainty_method: str, true_data: torch.Tensor, predicted_data: torch.Tensor,
                 causal_matrix: torch.Tensor,
                 predicted_data_std: Optional[torch.Tensor] = None,
                 causal_matrix_std: Optional[torch.Tensor] = None,
                 representations: Optional[torch.Tensor] = None):
        # Check that uncertainty_method is valid
        if uncertainty_method not in ['eval', 'monte_carlo', 'variational_inference']:
            raise ValueError(f"Unsupported uncertainty estimation method: {uncertainty_method}")

        # Initialize the attributes
        self.uncertainty_method: str = uncertainty_method
        self.true_data: torch.Tensor = true_data.detach().cpu()
        self.predicted_data: torch.Tensor = predicted_data.detach().cpu()

        # Set predicted_data_std to zeros if it is not provided
        self.predicted_data_std: torch.Tensor = torch.zeros_like(self.predicted_data) \
            if predicted_data_std is None else predicted_data_std.detach().cpu()

        # Ensure that predicted_data and predicted_data_std have the same shape
        assert self.predicted_data.shape == self.predicted_data_std.shape

        self.causal_matrix: torch.Tensor = causal_matrix.detach().cpu()

        # Set causal_matrix_std to zeros if it is not provided
        self.causal_matrix_std: torch.Tensor = torch.zeros_like(self.causal_matrix) \
            if causal_matrix_std is None else causal_matrix_std.detach().cpu()

        # Ensure that causal_matrix and causal_matrix_std have the same shape
        assert self.causal_matrix.shape == self.causal_matrix_std.shape

        # Determine if the model is history-dependent
        self.is_history_dependent: bool = len(self.causal_matrix.shape) == 4

        self.representations = representations  # (batch_size, num_variables, embedding_dim)

    def __repr__(self) -> str:
        """
        Returns a string representation of the ModelResult object.
        """
        return (f"{self.__class__.__name__}(uncertainty_method={self.uncertainty_method}, "
                f"true_data={self.true_data}, predicted_data={self.predicted_data}, "
                f"causal_matrix={self.causal_matrix}, "
                f"predicted_data_std={self.predicted_data_std}, "
                f"causal_matrix_std={self.causal_matrix_std})")


class TrainResult:
    def __init__(self, model: nn.Module, train_losses: List[float], test_losses: List[float]):
        self.n_parameters = count_parameters(model)
        self.train_losses: List[float] = train_losses
        self.test_losses: List[float] = test_losses


class EvaluationResult:
    def __init__(self,
                 dataset: Dataset,
                 model_config: ModelConfig,
                 model_result: ModelResult,
                 train_config: TrainConfig,
                 train_result: TrainResult
                 ):
        self.dataset = dataset
        self.model_config = model_config
        self.model_result = model_result
        self.train_config = train_config
        self.train_result = train_result