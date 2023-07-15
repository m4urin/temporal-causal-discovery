from typing import Any, Type

import numpy as np
import torch
from hyperopt import hp
from hyperopt.pyll import scope

from src.utils.pretty_printing import dict_to_url
from src.utils.pytorch import count_parameters


class TrainConfig:
    def __init__(self,
                 learning_rate: float,
                 num_epochs: int,
                 optimizer: type,
                 weight_decay: float,
                 test_size: int):
        self.learning_rate: float = learning_rate
        self.num_epochs: int = num_epochs
        self.optimizer = optimizer
        self.optimizer_name: str = optimizer.__name__
        self.weight_decay: float = weight_decay
        self.test_size = test_size

    @staticmethod
    def get_hp_space() -> dict[str, Any]:
        return {
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-3)),
            'num_epochs': scope.int(hp.quniform('num_epochs', 1000, 5000, 1000)),
            'optimizer': hp.choice('optimizer', [torch.optim.AdamW]),
            'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-2)),
            'test_size': 0.3
        }

    def __str__(self):
        return f"TrainConfig(learning_rate={self.learning_rate}, num_epochs={self.num_epochs}, " \
               f"optimizer={self.optimizer_name}, weight_decay={self.weight_decay}, test_size={self.test_size})"


class ModelConfig:
    def __init__(self, name: str, model_type, **model_kwargs):
        self.name = name
        self.url = ', '.join([name, dict_to_url(model_kwargs)])
        self.num_params: int = -1
        self.model_kwargs = model_kwargs
        self.model_type = model_type

    def instantiate_model(self):
        model = self.model_type(**self.model_kwargs)
        self.num_params = count_parameters(model)
        #print(f"Model {model.name()}, receptive_field={model.receptive_field}, num_parameters={self.num_params}")
        return model

    def __str__(self):
        return f"ModelConfig(model={self.name}, {dict_to_url(self.model_kwargs)})"

    def __repr__(self):
        return str(self)

