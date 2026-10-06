"""Stage 9.3: evaluation metrics — accuracy, precision, recall, F1, ROC-AUC,
EER, and expected calibration error (ECE)."""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)


def equal_error_rate(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob > lo) & (y_prob <= hi)
        if m.sum() == 0:
            continue
        acc = (y_true[m] == (y_prob[m] > 0.5)).mean()
        conf = y_prob[m].mean()
        ece += (m.sum() / len(y_prob)) * abs(acc - conf)
    return float(ece)


def compute_metrics(y_true, y_prob):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["eer"] = equal_error_rate(y_true, y_prob)
    else:
        out["roc_auc"] = float("nan")
        out["eer"] = float("nan")
    return out