import numpy as np
import torch
import torch.nn as nn
from hyperopt import hp
from hyperopt.pyll import scope

from src.models.modules.depthwise_separable_tcn import DepthwiseSeparableTCN
from src.models.temporal_causal_model import TemporalCausalModel
from src.utils.eval import monte_carlo_dropout
from src.utils.model_outputs import ModelOutput, ModelResult
from src.utils.progress import iter_batched


class NAVARRecurrentOutput(ModelOutput):
    def __init__(self, predictions: torch.Tensor, losses: torch.Tensor, contributions: torch.Tensor):
        """
        A class to store the output of a NAVAR model.

        Parameters:
        -----------
        predictions : torch.Tensor
            The model's predictions.
            Shape: (batch_size, num_var, sequence_length)
        losses : torch.Tensor
            The model's losses.
            Shape: (batch_size,)
        contributions : torch.Tensor
            The contributions of each variable to the predictions.
            Shape: (batch_size, num_var, sequence_length)
        """
        super().__init__(predictions, losses)
        self.contributions = contributions


class NAVARRecurrent(TemporalCausalModel):
    def __init__(self,
                 num_variables: int,
                 kernel_size: int,
                 n_layers: int,
                 n_repeats: int,
                 hidden_dim: int,
                 lambda1: float,
                 dropout: float,
                 **kwargs):
        """
        Neural Additive Vector AutoRegression (NAVAR) model using a
        Temporal Convolutional Network (TCN) with grouped convolutions.

        Transforms an input Tensor to Tensors 'prediction' and 'contributions':
        (batch_size, num_nodes, time_steps)
        -> (batch_size, num_nodes, time_steps), (batch_size, num_nodes, num_nodes, time_steps)

        Args:
            num_variables:
                The number of variables / time series (N)
            kernel_size:
                Kernel_size used by the TCN
            n_layers:
                Number of layers used by the TCN (!! for every layer, the tcn creates 2
                convolution layers with a residual connection)
            hidden_dim: int
                Dimensions of the hidden layers
            dropout: float
                Dropout probability of units in hidden layers
        """
        super().__init__(num_variables, receptive_field=-1)
        block1 = DepthwiseSeparableTCN(
            in_channels=num_variables,
            out_channels=num_variables * hidden_dim,
            kernel_size=kernel_size,
            n_blocks=1,  # no dilations
            n_layers_per_block=n_layers,
            groups=num_variables,
            dropout=dropout)

        block2 = DepthwiseSeparableTCN(
            in_channels=num_variables * hidden_dim,
            out_channels=num_variables * hidden_dim,
            kernel_size=kernel_size,
            n_blocks=1,  # no dilations
            n_layers_per_block=n_layers,
            groups=num_variables,
            dropout=dropout)

        contributions = nn.Conv1d(
            in_channels=num_variables * hidden_dim,
            out_channels=num_variables * num_variables,
            kernel_size=1,
            groups=num_variables)
        contributions.weight.data.normal_(0, 0.01)

        self.network = nn.Sequential(tcn, contributions)

        self.biases = nn.Parameter(torch.empty(1, num_variables, 1))
        self.biases.data.fill_(0.0001)
        self.lambda1 = lambda1

    @classmethod
    def get_hp_space(cls):
        return {
            'model_type': cls,
            'kernel_size': hp.choice('kernel_size', [2, 3, 5, 8]),
            'n_layers': scope.int(hp.quniform('n_layers', 1, 2, 1)),
            'hidden_dim': scope.int(hp.quniform('hidden_dim', 16, 64, 16)),
            'lambda1': hp.loguniform('lambda1', np.log10(0.01), np.log10(1.0)),
            'dropout': hp.quniform('dropout', 0, 0.5, 0.05),
        }

    def forward(self, x: torch.Tensor, max_batch_size: int = None) -> NAVAROutput:
        """
        x: Tensor of size (batch_size, num_variables, sequence_length)
        """
        batch_size = x.shape[0]

        # contributions is tensor of size (batch_size, num_nodes * num_nodes, sequence_length)
        if max_batch_size is None:
            # tensor of size (batch_size, num_nodes * num_nodes, sequence_length)
            contributions = self.network(x)
        else:
            assert max_batch_size > 0, 'batch_size must be larger greater or equal to 1.'
            # tensor of size (batch_size, num_nodes * num_nodes, sequence_length)
            contributions = torch.cat([self.network(_x) for _x in iter_batched(x, max_batch_size)], dim=0)

        # contributions: (bs, num_nodes, num_nodes, sequence_length)
        contributions = contributions.view(batch_size, self.num_variables, self.num_variables, -1)

        # predictions: (bs, num_nodes, time_steps)
        predictions = contributions.sum(dim=1) + self.biases

        regression_loss = ((predictions[..., :-1] - x[..., 1:]) ** 2).mean(dim=(1, 2))

        regularization_loss = contributions.abs().sum(dim=(1, 2)).mean(dim=-1)
        losses = regression_loss + (self.lambda1 / self.num_variables) * regularization_loss

        return NAVAROutput(predictions=predictions, losses=losses, contributions=contributions)

    def get_result(self, x: torch.Tensor) -> ModelResult:
        with torch.no_grad():
            model_output: NAVAROutput = monte_carlo_dropout(self, x)
            causal_matrix_data = model_output.contributions.std(dim=-1)
            return ModelResult(
                uncertainty_method='monte_carlo',
                true_data=x,
                predicted_data=model_output.predictions.mean(dim=0),
                predicted_data_std=model_output.predictions.std(dim=0),
                causal_matrix=causal_matrix_data.mean(dim=0),
                causal_matrix_std=causal_matrix_data.std(dim=0))

