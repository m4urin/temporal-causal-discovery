import torch
from typing import List, Any

from torch import nn

from src.models.config import ModelConfig, TrainConfig
from src.utils import concat_dicts
from src.utils.pytorch import count_parameters


class ModelOutput:
    def __init__(self, x_train=None, x_pred=None, mu_pred=None, std_pred=None, x_true=None, attn=None,
                 attn_instantaneous=None, attn_external_variables=None, regression_loss=None, kl_loss=None,
                 attn_loss=None, embeddings=None, embeddings_external_variables=None, uncertainty_method='none',
                 causal_matrix=None, causal_matrix_std=None):
        """
        Represents the output of a model.

        Args:
            x_train (torch.Tensor): The input of the model.
            x_pred (torch.Tensor): The prediction of the model.
            mu_pred (torch.Tensor): The mean of the prediction (optional).
            std_pred (torch.Tensor): The standard deviation of the prediction (optional).
            x_true (torch.Tensor): The true values of the dataset (optional).
            attn (torch.Tensor): Attention weights (optional).
            attn_instantaneous (torch.Tensor): Instantaneous attention weights (optional).
            attn_external_variables (torch.Tensor): Attention weights for external variables (optional).
            regression_loss (torch.Tensor): The regression loss for x (optional).
            kl_loss (torch.Tensor): The Kullback-Leibler divergence loss for variational layers (optional).
            attn_loss (torch.Tensor): The regularization loss for the attentions (optional).
            embeddings (torch.Tensor): The embeddings of the variables (optional).
            embeddings_external_variables (torch.Tensor): The embeddings of the external variables (optional).
            causal_matrix (torch.Tensor): Interpretations of the causal matrix
            causal_matrix_std (torch.Tensor): Variance in the interpretations of the causal matrix
        """
        # Check that uncertainty_method is valid
        if uncertainty_method not in ['eval', 'monte_carlo', 'variational', 'none']:
            raise ValueError(f"Unsupported uncertainty estimation method: {uncertainty_method}")

        self.x_train = x_train
        self.x_pred = x_pred
        self.mu_pred = mu_pred
        self.std_pred = std_pred
        self.x_true = x_true
        self.attn = attn
        self.attn_instantaneous = attn_instantaneous
        self.attn_external_variables = attn_external_variables
        self.regression_loss = regression_loss
        self.kl_loss = kl_loss
        self.attn_loss = attn_loss
        self.embeddings = embeddings
        self.embeddings_external_variables = embeddings_external_variables
        self.uncertainty_method = uncertainty_method
        self.causal_matrix = causal_matrix
        self.causal_matrix_std = causal_matrix_std

    def overwrite(self, x=None, mu=None, std=None, x_true=None, attn=None, attn_instantaneous=None,
                  attn_external_variables=None, loss=None, embeddings=None, embeddings_external_variables=None,
                  uncertainty_method=None, causal_matrix=None):
        """
        Creates a new ModelOutput.

        Parameters:
            x (optional): New value for the 'x' attribute.
            mu (optional): New value for the 'mu' attribute.
            std (optional): New value for the 'std' attribute.
            x_true (optional): New value for the 'x_true' attribute.
            attn (optional): New value for the 'attn' attribute.
            attn_instantaneous (optional): New value for the 'attn_instantaneous' attribute.
            attn_external_variables (optional): New value for the 'attn_external_variables' attribute.
            loss (optional): New value for the 'loss' attribute.
            embeddings (optional): New value for the 'embeddings' attribute.
            embeddings_external_variables (optional): New value for the 'embeddings_external_variables' attribute.
            uncertainty_method (optional): New value for the 'uncertainty_method' attribute.
            causal_matrix:..
        """
        new_params = {k: v for k, v in locals().items() if v is not None and k != 'self'}
        params = self.dict()
        for k, v in new_params.items():
            params[k] = v
        return ModelOutput(**params)

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
        loss = self.regression_loss
        if self.kl_loss is not None:
            loss += self.kl_loss
        if self.attn_loss is not None:
            loss += self.attn_loss

        if keep_batch:
            return loss
        else:
            return loss.sum()  # sum all losses from various batches

    def get_regression_loss(self):
        return self.regression_loss.sum()

    def add_mu_std(self, outputs: list):
        x = torch.cat([o.x_pred for o in outputs], dim=0)
        mu = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return self.overwrite(mu=mu, std=std, uncertainty_method='monte_carlo')

    def monte_carlo(self):
        data = {}
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                if k == 'x_pred' and (self.mu_pred is None or self.std_pred is None):
                    data['mu_pred'] = v.mean(dim=0, keepdim=True)
                    data['std_pred'] = v.std(dim=0, keepdim=True)
                if k == 'causal_matrix' and self.causal_matrix_std is None:
                    data['causal_matrix_std'] = v.std(dim=0, keepdim=True)
                data[k] = v.mean(dim=0, keepdim=True)
            elif not k.startswith('__'):
                data[k] = v
        return ModelOutput(**data)

    def dict(self):
        return {k: v for k, v in self.items()}

    def items(self):
        """
        Yields the attribute names and their corresponding tensor values.

        Yields:
            Tuple
                The attribute name and its corresponding tensor value.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor) or isinstance(attr, str):
                yield attr_name, attr

    def cpu(self):
        """
        Moves all tensors to the CPU.
        """
        for key in dir(self):
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.cpu())
        return self

    def detach(self):
        """
        Detaches every tensor from the computation graph.
        """
        for key in dir(self):
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.detach())
        return self

    def cuda(self):
        """
        Moves all tensors to the GPU.
        """
        for key in dir(self):
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.cuda())
        return self

    def __repr__(self):
        """
        Returns a string representation of the ModelOutput instance.

        Returns:
            str: The string representation of the ModelOutput instance.
        """
        attributes = []
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                attributes.append(f"\t{k} = {v.shape}")
            elif isinstance(v, str):
                attributes.append(f"\t{k} = {v}")
        attributes_str = "\n".join(attributes)
        return f"ModelOutput(\n{attributes_str}\n)"

    def __str__(self):
        return repr(self)


def cat_model_output(outputs: List[ModelOutput], dim: int):
    tensors = concat_dicts([o.dict() for o in outputs])
    tensors = {k: torch.cat(v, dim=dim) for k, v in tensors.items()}
    return ModelOutput(**tensors)


class TrainResult:
    def __init__(self, model: nn.Module,
                 train_losses: List[float], test_losses: List[float],
                 train_losses_true: List[float], test_losses_true: List[float],
                 auroc_scores: List[dict[str, Any]],
                 test_every: int, training_time: float):
        self.n_parameters = count_parameters(model)
        self.train_losses: List[float] = train_losses
        self.test_losses: List[float] = test_losses
        self.train_losses_true: List[float] = train_losses_true
        self.test_losses_true: List[float] = test_losses_true
        self.aucroc_scores: List[dict[str, Any]] = auroc_scores
        self.test_every = test_every
        self.training_time = training_time


class EvaluationResult:
    def __init__(self,
                 dataset,
                 model_config: ModelConfig,
                 model_output: ModelOutput,
                 train_config: TrainConfig,
                 train_result: TrainResult
                 ):
        self.dataset = dataset
        self.model_config = model_config
        self.model_output = model_output
        self.train_config = train_config
        self.train_result = train_result


