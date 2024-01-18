import torch
from torch import nn

from src.models.TAMCaD import TAMCaD
from scripts.old.NAVAR import NAVAR
from src.training.result import Result
from src.utils import count_parameters, receptive_field


class TemporalCausalModel(nn.Module):
    def __init__(self, model, tcn_architecture, **kwargs):
        super().__init__()
        # Initialize the causal model based on the 'model' argument
        if model == 'NAVAR':
            self.causal_model = NAVAR(**tcn_architecture, **kwargs)
        elif model == 'TAMCaD':
            self.causal_model = TAMCaD(**tcn_architecture, **kwargs)
        else:
            raise ValueError("Invalid model type specified.")

        self.n_params = count_parameters(self)  # Count number of parameters
        self.receptive_field = receptive_field(**tcn_architecture)  # Calculate receptive field size

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_ensembles * n_variables * 3, sequence_length).

        Returns:
            dict: Dictionary containing the prediction and contributions.
        """
        return self.causal_model(x)

    def compute_loss(self, y_true, **kwargs):
        return self.causal_model.compute_loss(y_true.unsqueeze(2), **kwargs)

    def analysis(self, x) -> Result:
        # Store current training mode and set model to eval mode for analysis
        mode = self.training
        self.eval()

        # Perform analysis without gradient computation
        with torch.no_grad():
            result = self.causal_model.analysis(**self.causal_model(x))

        # Revert to original mode after analysis
        self.train(mode)

        return result


if __name__ == '__main__':
    default_space = {
        'model': 'NAVAR',
        'hidden_dim': 32,
        'dropout': 0.2,
        'lambda1': 0.1,
        #'beta': 0.01,
        #'n_heads': None,
        #'attention_activation': 'softmax',
        'recurrent': False,
        'weight_sharing': False,
        'tcn_architecture': {'kernel_size': 2, 'n_blocks': 3, 'n_layers': 2},
        #'instantaneous': True
    }
    model = TemporalCausalModel(n_datasets=2, n_ensembles=5, n_variables=3, **default_space)

    x = torch.randn(1, 2, 5, 9, 100)
    y = torch.randn(1, 2, 3, 86)

    model_output = model(x)
    result = model.analysis(x)
    loss = model.compute_loss(y, **model_output)

    print(loss)
    print({k: v.size() for k, v in model_output.items()})
    print({k: v.size() for k, v in result.__dict__.items()})
