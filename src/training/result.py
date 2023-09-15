import torch


class Result:
    """Stores the result of a causal model.

    This class holds the prediction as well as different types of causal matrices
    and their standard deviations.

    Attributes:
        prediction (torch.Tensor): The prediction tensor.
        causal_matrix (torch.Tensor): The causal matrix tensor.
        causal_matrix_std (torch.Tensor): The standard deviation of the causal matrix tensor.
        temporal_causal_matrix (torch.Tensor): The temporal causal matrix tensor.
        temporal_causal_matrix_std (torch.Tensor): The standard deviation of the temporal causal matrix tensor.
    """
    def __init__(self, prediction: torch.Tensor, causal_matrix: torch.Tensor,
                 causal_matrix_std: torch.Tensor, temporal_causal_matrix: torch.Tensor,
                 temporal_causal_matrix_std: torch.Tensor):
        """
        Initializes the Result object with the given tensors.

        Args:
            prediction (torch.Tensor): The prediction tensor.
            causal_matrix (torch.Tensor): The causal matrix tensor.
            causal_matrix_std (torch.Tensor): The standard deviation of the causal matrix tensor.
            temporal_causal_matrix (torch.Tensor): The temporal causal matrix tensor.
            temporal_causal_matrix_std (torch.Tensor): The standard deviation of the temporal causal matrix tensor.
        """
        self.prediction = prediction
        self.causal_matrix = causal_matrix
        self.causal_matrix_std = causal_matrix_std
        self.temporal_causal_matrix = temporal_causal_matrix
        self.temporal_causal_matrix_std = temporal_causal_matrix_std

    @property
    def args(self):
        return {
            'prediction': self.prediction,
            'causal_matrix': self.causal_matrix,
            'causal_matrix_std': self.causal_matrix_std,
            'temporal_causal_matrix': self.temporal_causal_matrix,
            'temporal_causal_matrix_std': self.temporal_causal_matrix_std
        }
