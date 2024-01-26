from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class Dataset:
    """
    Dataclass to hold preprocessed data.

    This dataclass stores the preprocessed data, the mean, and optionally the ground truth.

    Attributes:
        data_dir (str): Directory where the dataset is stored.
        name (str): Identifier for the dataset.
        data (torch.Tensor): Tensor holding the normalized data.
        data_mean (Optional[torch.Tensor]): Mean of the data, also normalized. Defaults to None.
        ground_truth (Optional[torch.Tensor]): Ground truth tensor if available. Defaults to None.
    """
    data_dir: str
    name: str
    data: torch.Tensor
    data_mean: Optional[torch.Tensor] = field(default=None)
    ground_truth: Optional[torch.Tensor] = field(default=None)

    def as_dict(self):
        return {
            'name': self.name,
            'data': self.data[:, 0]
        }


@dataclass
class ModelConfig:
    """Configuration dataclass for models.

    Attributes:
        kernel_size (int): Size of the convolutional kernel.
        n_blocks (int): Number of blocks in the TCN.
        n_layers (int): Number of layers per block.
        recurrent (bool): Whether the TCN is recurrent.
        weight_sharing (bool): Whether to share weights across layers int he TCN.
        use_padding (bool): Whether to use padding in the TCN to get the same input and output size.
        dropout (float): Dropout rate.
        hidden_dim (int): Dimension of hidden layers.
        model_name (str): The name of the model.
        attention_activation (str): Type of attention activation to use.
        lambda1 (float): Lambda1 hyperparameter.
        beta (float): Beta hyperparameter.
        n_heads (int, Optional): Number of attention heads when using scaled dot product.
    """
    kernel_size: int
    n_blocks: int
    n_layers: int
    recurrent: bool
    weight_sharing: bool
    use_padding: bool
    dropout: float
    hidden_dim: int
    model_name: str
    attention_activation: str
    lambda1: float
    beta: float
    n_heads: int = None


@dataclass
class TrainConfig:
    """
    Configuration for training a model.

    Attributes:
        test_size (float): The proportion of the dataset to use as the test set.
        lr (float): Learning rate.
        epochs (int): Number of training epochs.
        weight_decay (float): Weight decay factor.
    """
    test_size: float
    lr: float
    epochs: int
    weight_decay: float


@dataclass
class HyperoptConfig:
    """
    Configuration for hyperparameter optimization.

    Attributes:
        max_evals (int): Maximum number of evaluations during hyperparameter tuning.
        n_datasets (int): Number of datasets per evaluation.
        n_ensemble (int): Number of ensemble model per dataset.
    """
    max_evals: int
    n_datasets: int
    n_ensemble: int


@dataclass
class ExperimentConfig:
    """
    Configuration for an experiment.

    Attributes:
        dataset (Dataset): The Dataset object.
        model (ModelConfig): The ModelConfig object.
        training (TrainConfig): The TrainConfig object.
        hyperopt (HyperoptConfig): The HyperoptConfig object.
    """
    dataset: Dataset
    model: ModelConfig
    training: TrainConfig
    hyperopt: HyperoptConfig


@dataclass
class CausalModelResult:
    """
    Stores the result of a causal model.

    Attributes:
        prediction (torch.Tensor): Tensor containing the prediction.
        causal_matrix (torch.Tensor): Tensor representing the causal matrix.
        causal_matrix_std (torch.Tensor): Standard deviation of the causal matrix.
        temporal_causal_matrix (torch.Tensor): Tensor for the temporal causal matrix.
        temporal_causal_matrix_std (torch.Tensor): Standard deviation of the temporal causal matrix.
    """
    prediction: torch.Tensor
    causal_matrix: torch.Tensor
    causal_matrix_std: torch.Tensor
    temporal_causal_matrix: torch.Tensor
    temporal_causal_matrix_std: torch.Tensor
