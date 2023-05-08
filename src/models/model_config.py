import numpy as np
import torch
from hyperopt import hp
from hyperopt.pyll import scope
from torch import nn

from src.utils.pretty_printing import dict_to_url
from src.utils.pytorch import count_parameters


class TrainConfig:
    def __init__(self,
                 learning_rate: float,
                 num_epochs: int,
                 optimizer: type,
                 weight_decay: float,
                 val_proportion: float,
                 **kwargs):
        self.learning_rate: float = learning_rate
        self.num_epochs: int = num_epochs
        self.optimizer = optimizer
        self.optimizer_name: str = optimizer.__name__
        self.weight_decay: float = weight_decay
        self.val_proportion = val_proportion

    @staticmethod
    def get_hp_space(val_proportion: float = 0.0):
        return {
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-3)),
            'num_epochs': scope.int(hp.quniform('num_epochs', 1000, 5000, 1000)),
            'optimizer': hp.choice('optimizer', [torch.optim.AdamW]),
            'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-2)),
            'val_proportion': val_proportion
        }


class ModelConfig:
    def __init__(self, model_type, **model_kwargs):
        self.name = f"{model_type.__name__}({dict_to_url(model_kwargs)})"
        self.url = dict_to_url({'model': model_type, **{k: model_kwargs[k]
                                                        for k in model_type.get_hp_space().keys()
                                                        if k != "model_type"}})
        self.model_type = model_type
        self.num_params: int = -1
        self.model_kwargs = model_kwargs

    def instantiate_model(self):
        model = self.model_type(**self.model_kwargs)
        self.num_params = count_parameters(model) if isinstance(model, nn.Module) else 0
        return model

    def __str__(self):
        return f"ModelConfig[{self.name}]"

    def __repr__(self):
        return str(self)
