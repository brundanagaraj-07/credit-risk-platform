"""
Evaluation metrics for the imbalanced binary classification task.

ROC-AUC alone is a poor headline metric here because negatives outnumber
positives ~11:1 -- a model that is right on the majority class inflates
ROC-AUC while missing most actual defaulters. We therefore always report
PR-AUC (average precision) and precision/recall/F1 at the deployed decision
threshold alongside it.
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
    }


def find_best_threshold(y_true, y_prob) -> float:
    """Pick the probability threshold that maximises F1 -- used to suggest
    (not force) an operating point for the risk-band cutoffs."""
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        f1 = f1_score(y_true, (np.asarray(y_prob) >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)
