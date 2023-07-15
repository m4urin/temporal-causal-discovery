import torch
from torch import nn
from torch.nn import Conv1d

from src.models.model_outputs import ModelOutput, cat_model_output
from src.models.modules.interpret.interpreter import Interpreter
from src.models.modules.temporal.external_variables import ExternalVariables
from src.models.modules.temporal.temporal_block import TemporalBlock
from src.models.modules.temporal.temporal_module import TemporalModule
from src.models.modules.tcn.base_tcn import BaseTCN
from src.models.modules.tcn.tcn_rec import RecTCN
from src.models.modules.tcn.tcn_ws import WSTCN


class TemporalCausalModel(TemporalModule):
    def __init__(self,
                 num_variables,
                 hidden_dim,
                 receptive_field: dict[str, int],
                 lambda1: float,
                 beta1: float,
                 dropout,
                 max_sequence_length: int,
                 use_recurrent=False,
                 use_weight_sharing=False,
                 use_variational_layer=False,
                 use_instantaneous_predictions=False,
                 use_attentions=False,
                 use_navar=False,
                 num_external_variables: int = 0,
                 num_heads: int = None):
        n_blocks = receptive_field['n_blocks']
        n_layers_per_block = receptive_field['n_layers_per_block']
        kernel_size = receptive_field['kernel_size']

        super().__init__(num_variables, num_variables, num_variables,
                         receptive_field=(2 ** n_blocks - 1) * n_layers_per_block * (kernel_size - 1) + 1)
        hidden_dim *= num_variables

        self.use_recurrent = use_recurrent
        self.use_weight_sharing = use_weight_sharing
        self.use_variational_layer = use_variational_layer
        self.interpreter = 'navar' if use_navar else 'attention'
        self.num_external_variables = num_external_variables
        self.use_instantaneous_predictions = use_instantaneous_predictions

        name = [('Var', use_variational_layer), ('WS', use_weight_sharing), ('Rec', use_recurrent),
                (f'Ext{num_external_variables}', num_external_variables > 0), ('NAVAR', use_navar),
                ('Attn', use_attentions)]
        self.name = "-".join([tag for tag, select in name if select])

        self.lambda1 = lambda1
        self.beta1 = beta1

        if use_weight_sharing:
            self.tcn = WSTCN(num_variables, hidden_dim, hidden_dim, kernel_size, max_sequence_length,
                             n_blocks, n_layers_per_block, num_variables, dropout, use_recurrent=use_recurrent)
        elif use_recurrent:
            self.tcn = RecTCN(num_variables, hidden_dim, hidden_dim, kernel_size,
                              n_blocks, n_layers_per_block, num_variables, dropout)
        else:
            self.tcn = BaseTCN(num_variables, hidden_dim, hidden_dim, kernel_size,
                               n_blocks, n_layers_per_block, num_variables, dropout)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.external_variables = None
        if num_external_variables > 0:
            self.external_variables = ExternalVariables(hidden_dim, num_variables, num_external_variables,
                                                        max_sequence_length)
            hidden_dim = (num_variables + num_external_variables) * (hidden_dim // num_variables)
            num_variables += num_external_variables

        self.interpreter = Interpreter(hidden_dim, num_variables, num_external_variables,
                                       use_navar, use_attentions, use_variational_layer,
                                       use_instantaneous_predictions, num_heads)

    def forward_mini_batch(self, x: torch.Tensor, x_true: torch.Tensor = None, to_cpu=False) -> ModelOutput:
        y = self.tcn(x)
        y = self.relu(y)
        if self.external_variables is not None:
            y = self.external_variables(y)
        y = self.dropout(y)
        result: ModelOutput = self.interpreter(y)

        # Fix the length of the predictions.
        # The model required the full length because of the instantaneous predictions.
        result.x_pred = result.x_pred[..., :-1]
        if result.mu_pred is not None:
            result.mu_pred = result.mu_pred[..., :-1]
        if result.std_pred is not None:
            result.std_pred = result.std_pred[..., :-1]

        result.x_train = x[..., :-1]

        # losses
        if x_true is None:
            result.regression_loss = (result.x_pred - x[..., 1:]).pow(2).mean(dim=(1, 2))
        else:
            result.regression_loss = (result.x_pred - x_true[..., 1:]).pow(2).mean(dim=(1, 2))

        if result.kl_loss is not None:
            result.kl_loss *= self.beta1
        if result.attn_loss is not None:
            result.attn_loss *= self.lambda1

        # weight sharing embeddings
        if isinstance(self.tcn, WSTCN):
            result.embeddings = self.tcn.get_embeddings()

        # embeddings of external variables
        if self.external_variables is not None:
            result.embeddings_external_variables = self.external_variables.get_embeddings()

        if to_cpu:
            result = result.cpu()
        return result

    def forward(self, x: torch.Tensor, x_true: torch.Tensor = None, max_batch_size: int = None, to_cpu=False) -> ModelOutput:
        batch_size = x.size(0)

        if max_batch_size is None or batch_size <= max_batch_size:
            return self.forward_mini_batch(x, x_true, to_cpu)

        iterations = batch_size // max_batch_size
        if batch_size % max_batch_size != 0:
            iterations += 1

        all_outputs = []
        for i in range(iterations):
            all_outputs.append(self.forward_mini_batch(x[i * max_batch_size:(i + 1) * max_batch_size], to_cpu=to_cpu))

        # merge to one model_output
        return cat_model_output(all_outputs, dim=0)

    def forward_eval(self, x: torch.Tensor, x_true: torch.Tensor = None, max_batch_size: int = None) -> ModelOutput:
        """ Same as forward(), but with torch.no_grad and run in eval() mode. """
        with torch.no_grad():
            mode = self.training
            self.eval()
            model_output = self.forward(x, x_true, max_batch_size, to_cpu=False)
            self.train(mode)
            return model_output

    def monte_carlo(self, x: torch.Tensor, x_true: torch.Tensor = None, samples: int = 100, max_batch_size: int = None) -> ModelOutput:
        with torch.no_grad():
            mode = self.training
            self.train()
            result = self.forward(x.expand(samples, -1, -1),
                                  x_true.expand(samples, -1, -1), max_batch_size, to_cpu=False)
            result = result.monte_carlo()
            result.uncertainty_method = 'monte_carlo'
            self.train(mode)
            return result
