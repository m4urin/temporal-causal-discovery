import math
import torch
from matplotlib import pyplot as plt


def roc_auc_score(labels: torch.Tensor, y_pred: torch.Tensor):
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must contain only 0's and 1's.")

    batch_size = labels.size(0)

    labels = labels.reshape(batch_size, -1)  # binary True/False
    y_pred = y_pred.reshape(batch_size, -1)

    # Sort scores and corresponding truth values
    desc_score_indices = torch.argsort(y_pred, descending=True)
    y_pred = torch.gather(y_pred, dim=-1, index=desc_score_indices)
    labels = torch.gather(labels, dim=-1, index=desc_score_indices)

    # Compute the number of positive and negative samples
    num_pos = torch.sum(labels, dim=-1).unsqueeze(-1)
    num_neg = labels.size(-1) - num_pos

    # Compute cumulative positives and negatives
    tps = torch.cumsum(labels, dim=-1).float()
    fps = torch.arange(1, labels.shape[-1] + 1, device=y_pred.device).float() - tps

    # Compute FPR, TPR
    fpr = fps / num_neg
    tpr = tps / num_pos

    # Compute AUC
    auc = torch.trapz(tpr, fpr, dim=-1)

    return auc, tpr, fpr


def soft_roc_auc_score(labels: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor, n_thresholds=500) -> tuple:
    """
    Compute the soft ROC AUC based on given predictions and their associated uncertainties.
    """
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must contain only 0's and 1's.")

    batch_size = labels.size(0)

    # Make sure the tensors are flat
    labels = labels.reshape(batch_size, -1, 1).bool()  # binary True/False
    y_mean = y_mean.reshape(batch_size, -1)
    y_std = y_std.reshape(batch_size, -1)

    n_positive = torch.sum(labels, dim=1)
    n_negative = labels.size(1) - n_positive

    thresholds = torch.cat([
        torch.linspace(-10, -2, (n_thresholds // 5) - 1, device=y_mean.device),
        torch.linspace(-2, 3, n_thresholds - 2 * (n_thresholds // 4), device=y_mean.device),
        torch.linspace(3, 11, (n_thresholds // 5) - 1, device=y_mean.device)
    ]).reshape(1, 1, -1)  # size (1, 1, n_thresholds)

    # cdf = 0.5 * (1 + erf) -> prob = 1 - (0.5 * (1 + erf)) = 0.5 - 0.5 * erf
    # erf is of size (bs, n_labels, n_thresholds)
    probabilities = 0.5 - 0.5 * torch.erf((thresholds - y_mean.unsqueeze(-1)) / (y_std.unsqueeze(-1) * math.sqrt(2)))

    tpr = torch.masked_fill(probabilities, ~labels, value=0).sum(dim=1) / n_positive
    fpr = torch.masked_fill(probabilities, labels, value=0).sum(dim=1) / n_negative

    zeros = torch.zeros(batch_size, 1, device=y_mean.device)
    ones = torch.ones(batch_size, 1, device=y_mean.device)

    tpr = torch.cat((zeros, tpr.flip(dims=[1]), ones), dim=-1)
    fpr = torch.cat((zeros, fpr.flip(dims=[1]), ones), dim=-1)

    # Compute AUC
    auc = torch.trapz(tpr, fpr)

    return auc, tpr, fpr


def main():
    bs, n = 2, 8
    y_true = torch.rand(bs, n, n)
    y_means = 0.7 * torch.rand(bs, n, n) + 0.3 * y_true
    y_true = (y_true > 0.5).float()
    y_stds = torch.rand(bs, n, n) + 0.001

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    auc, tpr, fpr = roc_auc_score(y_true, y_means)
    plt.plot(fpr[0], tpr[0], label=f'default (AUC={auc[0]:.2f})', linestyle='--')  # Plot the ROC curve for this method

    for c in [0.001, 0.2, 1.0, 5.0, 20.0]:
        auc, tpr, fpr = soft_roc_auc_score(y_true, y_means, y_stds * c)
        plt.plot(fpr[0], tpr[0], label=f'{c} (AUC={auc[0]:.2f})', )  # Plot the ROC curve for this method

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
