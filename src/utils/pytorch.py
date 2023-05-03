from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Counts the number of parameters in a PyTorch model.

    Args:
        model: A PyTorch model to count the parameters of.
        trainable_only: A boolean flag indicating whether to count only trainable parameters.
            If True, only parameters with requires_grad=True will be counted.

    Returns:
        An integer representing the total number of parameters in the model.
    """
    if trainable_only:
        # count only trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        # count all parameters, including non-trainable parameters
        total_params = sum(p.numel() for p in model.parameters())

    return total_params


def share_parameters(default_model: nn.Module, *model_list: nn.Module) -> None:
    """
    Shares the parameter tensors of a default model with a list of PyTorch models.

    Args:
        default_model: A PyTorch model whose parameters will be shared with other models.
        *model_list: A variable number of PyTorch models to share parameters with.

    Raises:
        ValueError: If the list of models is empty.

    Note:
        This function assumes that all models have the same architecture, i.e., the same
        set of named parameters. If the models have different architectures, sharing their
        parameters may result in unexpected behavior.
    """
    if len(model_list) == 0:
        raise ValueError("model_list must not be empty")

    # iterate over the named parameters of the default model
    for name, param in default_model.named_parameters():
        # share the parameter tensor between all the models in the list
        name = name.split('.')
        for model in model_list:
            target_param = model
            for n in name[:-1]:
                target_param = getattr(target_param, n)
            setattr(target_param, name[-1], param)
