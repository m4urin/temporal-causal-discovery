import torch


def NLL_loss(y_true, y_pred, log_var):
    """
    Calculate the gaussian negative log likelihood (NLL) loss.

    Args:
        y_true (torch.Tensor): The true target values.
        y_pred (torch.Tensor): The predicted target values.
        log_var (torch.Tensor): The logarithm of the predicted variance.

    Returns:
        torch.Tensor: The NLL loss value.
    """
    # Calculate the squared error term
    error = (y_pred - y_true).pow(2)
    # Calculate the NLL loss as the mean of log variance and the squared error term
    return torch.mean(log_var + error / (torch.exp(log_var) + 1e-8))


def DER_loss(y_true, y_pred, log_var, log_v, coeff):
    """
    Calculate the Deep Evidential Regression (DER) loss.

    Args:
        y_true (torch.Tensor): The true target values.
        y_pred (torch.Tensor): The predicted target values.
        log_var (torch.Tensor): The logarithm of the predicted variance.
        log_v (torch.Tensor): The uncertainty parameter tensor.
        coeff (float): Coefficient for regularizing the uncertainty parameter.

    Returns:
        torch.Tensor: The DER loss value.
    """
    # Calculate the squared error term with modified uncertainty parameter
    error = (1.0 + coeff * torch.exp(log_v)) * (y_pred - y_true).pow(2)
    # Calculate the DER loss as the mean of log variance and the modified error term
    return torch.mean(log_var + error.div(torch.exp(log_var)))


def NAVAR_regularization_loss(contributions, lambda1=0.2):
    """
    Calculate the regularization loss for NAVAR.

    This regularization encourages the model to learn sparse or zero contributions by penalizing
    the absolute mean of the contributions with a regularization coefficient lambda1.

    Args:
        contributions (torch.Tensor): The contributions to be regularized.
        lambda1 (float, optional): Regularization coefficient. Default is 0.2.

    Returns:
        torch.Tensor: The NAVAR regularization loss value.
    """
    # Calculate the absolute mean of contributions and scale by lambda1
    regularization_term = lambda1 * contributions.abs().mean()
    return regularization_term


def TAMCaD_regularization_loss(attentions, lambda1):
    """
    Calculate the regularization loss for TAMCaD.

    This regularization encourages the model to learn zero-attention to its future self by penalizing
    these attentions with a regularization coefficient lambda1.

    Args:
        attentions (torch.Tensor): The (positive) attentions to be regularized,
                                   the size is (batch_size, n_heads, n, 2*n, sequence_length)
        lambda1 (float, optional): Regularization coefficient. Default is 0.2.

    Returns:
        torch.Tensor: The TAMCaD regularization loss value.
    """
    # Calculate the absolute mean of contributions and scale by lambda1
    n = attentions.size(2)
    return lambda1 * attentions[..., range(n), range(n, 2 * n), :].mean()
