"""Discrimination, calibration and cost metrics for a binary default model."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def gini(y_true, p) -> float:
    """Gini coefficient, 2 * AUC - 1."""
    return float(2 * roc_auc_score(y_true, p) - 1)


def ks_statistic(y_true, p) -> float:
    """Maximum gap between the score CDFs of positives and negatives."""
    fpr, tpr, _ = roc_curve(y_true, p)
    return float(np.max(tpr - fpr))


def expected_cost(y_true, p, threshold: float, fn_cost: float, fp_cost: float) -> float:
    """Mean cost per client when predicting positive for ``p >= threshold``."""
    y = np.asarray(y_true).astype(bool)
    predicted = np.asarray(p) >= threshold
    false_negatives = np.sum(y & ~predicted)
    false_positives = np.sum(~y & predicted)
    return float((fn_cost * false_negatives + fp_cost * false_positives) / len(y))


def classification_metrics(y_true, p) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "gini": gini(y_true, p),
        "ks": ks_statistic(y_true, p),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
    }
