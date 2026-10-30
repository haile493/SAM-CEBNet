"""
Standardized Evaluation Metrics for 2D/3D Segmentation and Multi-Class / Multi-Label Classification.
"""

from typing import Dict, Union

import numpy as np
import torch


def compute_binary_segmentation_metrics(
    pred_mask: Union[torch.Tensor, np.ndarray],
    target_mask: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Computes Dice (F1), IoU (Jaccard), Precision, Recall (Sensitivity), and Specificity
    for binary 2D or 3D masks. Accepts PyTorch tensors or NumPy arrays.
    """
    if isinstance(pred_mask, np.ndarray):
        pred_t = torch.from_numpy(pred_mask)
    else:
        pred_t = pred_mask.detach()

    if isinstance(target_mask, np.ndarray):
        tgt_t = torch.from_numpy(target_mask)
    else:
        tgt_t = target_mask.detach()

    pred_bin = (pred_t.float() >= threshold).float().reshape(-1)
    tgt_bin = (tgt_t.float() >= 0.5).float().reshape(-1)

    tp = (pred_bin * tgt_bin).sum().item()
    fp = (pred_bin * (1.0 - tgt_bin)).sum().item()
    fn = ((1.0 - pred_bin) * tgt_bin).sum().item()
    tn = ((1.0 - pred_bin) * (1.0 - tgt_bin)).sum().item()

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "f1": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
    }


def compute_classification_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    multi_label: bool = False,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes standardized classification metrics (Accuracy, Macro F1, Weighted F1, Precision, Recall).
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    probs_np = probs.detach().cpu().numpy() if isinstance(probs, torch.Tensor) else np.asarray(probs)
    targets_np = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)

    if multi_label:
        preds = (probs_np >= threshold).astype(int)
        y_true = targets_np.astype(int)
    else:
        preds = np.argmax(probs_np, axis=1) if probs_np.ndim > 1 else (probs_np >= threshold).astype(int)
        y_true = targets_np.astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, preds, average="macro", zero_division=0)),
    }
