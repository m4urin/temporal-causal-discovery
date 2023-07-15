from sklearn import metrics
import numpy as np


def calculate_AUROC(score_matrix, ground_truth, ignore_self_links=False):
    """
    Calculates the Area Under Receiver Operating Characteristic Curve

    Args:
        score_matrix: ndarray
            ndarray containing scores for every potential causal link
        ground_truth: ndarray
            binary ndarrray containing ground truth causal links
        ignore_self_links: bool
            indicates whether we should ignore self-links (i.e. principal diagonal) in AUROC calculation
    Returns:
        aucroc: float
            Area Under Receiver Operating Characteristic Curve
    """
    score_matrix_flattened = score_matrix.cpu().flatten()
    ground_truth_flattened = ground_truth.cpu().flatten()
    if ignore_self_links:
        indices = np.arange(0, score_matrix_flattened.shape[0], int(np.round(np.sqrt(score_matrix_flattened.shape[0]))) + 1)
        score_matrix_flattened = np.delete(score_matrix_flattened, indices)
        ground_truth_flattened = np.delete(ground_truth_flattened, indices)
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth_flattened, score_matrix_flattened)

    aucroc = metrics.auc(fpr, tpr)

    return {
        'score': aucroc,
        'fpr':  fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'causal_matrix': score_matrix.cpu().numpy()
    }
