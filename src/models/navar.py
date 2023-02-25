import torch.nn as nn
import torch


class NAVAR(nn.Module):
    def __init__(self, num_nodes: int, kernel_size: int, n_layers: int,
                 hidden_dim: int, lambda1: float, dropout: float):
        """
        Neural Additive Vector AutoRegression (NAVAR) model
        Args:
            num_nodes: int
                The number of time series (N)
            hidden_dim: int
                Number of hidden units per layer
            kernel_size: int
                Maximum number of time lags considered (K)
            n_layers: int
                Number of hidden layers
            lambda1: float
                Lambda for regularization
            dropout:
                Dropout probability of units in hidden layers
        """
        super(NAVAR, self).__init__()
        self.num_nodes = num_nodes
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lambda1 = lambda1
        self.dropout = dropout

    def get_receptive_field(self):
        raise NotImplementedError('Not implemented yet')

    def get_loss(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        raise NotImplementedError('Not implemented yet')

    def forward(self, x: torch.Tensor, **kwargs):
        raise NotImplementedError('Not implemented yet')

    def evaluate(self, x: torch.Tensor, monte_carlo_dropout: int = None, **kwargs):
        raise NotImplementedError('Not implemented yet')


def test_model(model_constructor: type(NAVAR)):
    from src.utils import count_params
    batch_size = 1
    time_steps = 100
    num_nodes = 5

    model = model_constructor(num_nodes=num_nodes, kernel_size=3, n_layers=2,
                              hidden_dim=16, lambda1=0.1, dropout=0.1)

    x = torch.rand((batch_size, num_nodes, time_steps))
    y = torch.rand((batch_size, num_nodes, time_steps))

    result = model(x)
    loss = model.get_loss(x, y)
    evaluation = model.evaluate(x, monte_carlo_dropout=10, sample=False)

    print(model, '\n')
    print(f"n_parameters_per_node={count_params(model) // 5}")
    print(f"receptive_field={model.get_receptive_field()}")
    print(f"data size={x.size()}")
    print(f"loss={loss}")
    print("result=")
    for v in result:
        print(f"\t{v.size()}")
    print("eval=")
    for v in evaluation:
        print(f"\t{v.size()}")
