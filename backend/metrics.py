import numpy as np


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()

    return float((2.0 * intersection + eps) / (pred.sum() + gt.sum() + eps))