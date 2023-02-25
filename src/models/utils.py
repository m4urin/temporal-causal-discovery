import torch
from torch import nn


class View(nn.Module):
    def __init__(self, *view):
        super(View, self).__init__()
        self.view = view

    def forward(self, x: torch.Tensor):
        return x.view(*self.view)


class Addition(nn.Module):
    def __init__(self, param: nn.Parameter):
        super(Addition, self).__init__()
        self.param = param

    def forward(self, x: torch.Tensor):
        return x + self.param
