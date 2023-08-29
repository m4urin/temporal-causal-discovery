import math
import torch


def roc_auc_score(y_true: torch.Tensor, y_score: torch.Tensor):
    if not torch.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must contain only 0's and 1's.")

    n = y_true.size(1)
    y_true = y_true[..., :n]

    # Make sure the tensors are flat
    y_true = y_true.flatten()
    y_score = y_score.flatten()

    # Ensure tensors are on the same device
    device = y_score.device

    # Sort scores and corresponding truth values
    desc_score_indices = torch.argsort(y_score, descending=True)
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # Compute the number of positive and negative samples
    num_pos = torch.sum(y_true).float()
    num_neg = y_true.shape[0] - num_pos

    # Compute cumulative positives and negatives
    tps = torch.cumsum(y_true, dim=0).float()
    fps = torch.arange(1, y_true.shape[0] + 1, device=device).float() - tps

    # Compute FPR, TPR
    fpr = fps / num_neg
    tpr = tps / num_pos

    # Compute AUC
    auc = torch.trapz(tpr, fpr)

    return fpr, tpr, auc


def soft_roc_auc_score(y_true: torch.Tensor, y_score: torch.Tensor, y_std: torch.Tensor, n_samples=200) -> tuple:
    """
    Compute the soft ROC AUC based on given predictions and their associated uncertainties.
    """
    if not torch.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must contain only 0's and 1's.")

    n = y_true.size(1)
    y_true = y_true[:, :, :n]

    # Make sure the tensors are flat
    y_true = y_true.flatten()
    y_score = y_score.flatten()
    y_std = y_std.flatten()

    thresholds = torch.linspace(1.0/n_samples, 1.0 - 1.0/n_samples, n_samples-2, device=y_true.device)
    probabilities = compute_cdf(thresholds, y_score, y_std)

    # Compute the number of positive and negative samples
    num_pos = torch.sum(y_true).float()
    num_neg = y_true.shape[0] - num_pos

    y_true_bin = y_true > 0
    tps = torch.sum(1.0 - probabilities[y_true_bin], dim=0)
    fps = torch.sum(1.0 - probabilities[~y_true_bin], dim=0)

    tpr = tps / num_pos
    fpr = fps / num_neg

    sorted_indices = torch.argsort(fpr)
    tpr, fpr = tpr[sorted_indices], fpr[sorted_indices]

    tpr = torch.cat((torch.tensor([0], device=y_true.device), tpr, torch.tensor([1], device=y_true.device)))
    fpr = torch.cat((torch.tensor([0], device=y_true.device), fpr, torch.tensor([1], device=y_true.device)))

    # Compute AUC
    auc = torch.trapz(tpr, fpr)

    return fpr, tpr, auc


def compute_cdf(values: torch.Tensor, means: torch.Tensor, std_devs: torch.Tensor) -> torch.Tensor:
    """
    Compute the CDF using the Error Function for a given set of values, means, and standard deviations.
    """
    values = values[:, None]
    means = means[None, :]
    std_devs = std_devs[None, :]
    return 0.5 * (1 + torch.erf((values - means) / (std_devs * math.sqrt(2)))).transpose(0, 1)
