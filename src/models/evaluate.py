"""Cycle 3 — evaluation metrics per Master Architecture Document Pillar 4.

Accuracy alone is a trap under class imbalance (4.1) -- this module always
computes the fuller picture: per-class precision/recall/FPR, macro-F1,
PR-AUC and ROC-AUC (leading with PR-AUC per 4.2), and the confusion matrix.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# Classes with too few test-fold examples for a stable per-run recall
# estimate -- a single Heartbleed test row flips right/wrong and swings
# macro-recall by roughly +/-11 points on its own. Excluded from the
# regression-gated macro-recall figure; still reported individually.
LOW_CONFIDENCE_CLASSES = {"Heartbleed", "Infiltration"}


def _per_class_fpr(cm: np.ndarray) -> np.ndarray:
    """One-vs-rest false positive rate per class from a confusion matrix.

    sklearn has no direct multi-class FPR function, so this is computed by
    hand: for class c, FP = predicted c but not actually c; TN = neither
    predicted nor actually c.
    """
    total = cm.sum()
    fpr = np.zeros(cm.shape[0])
    for c in range(cm.shape[0]):
        fp = cm[:, c].sum() - cm[c, c]
        tn = total - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fpr[c] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return fpr


def aggregate_benign_fpr(y_true: np.ndarray, y_pred: np.ndarray, benign_id: int) -> float:
    """Binary FPR: fraction of true-BENIGN rows predicted as any attack
    class. The one metric well-defined on an all-BENIGN holdout (Monday),
    so this is what the day-based leakage smell test compares."""
    true_benign = y_true == benign_id
    if true_benign.sum() == 0:
        raise ValueError("no true-BENIGN rows to compute FPR against")
    false_positives = true_benign & (y_pred != benign_id)
    return float(false_positives.sum() / true_benign.sum())


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_map: dict,
) -> dict:
    """label_map: {class_name: class_id}. Returns a JSON-serializable dict."""
    id_to_label = {v: k for k, v in label_map.items()}
    class_ids = sorted(id_to_label)
    class_names = [id_to_label[i] for i in class_ids]

    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    fpr_per_class = _per_class_fpr(cm)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_ids, zero_division=0
    )

    y_true_onehot = label_binarize(y_true, classes=class_ids)
    pr_auc_per_class = average_precision_score(y_true_onehot, y_proba, average=None)
    try:
        roc_auc_per_class = roc_auc_score(y_true_onehot, y_proba, average=None, multi_class="ovr")
    except ValueError:
        # a class absent from y_true in this fold has an undefined ROC-AUC
        roc_auc_per_class = np.full(len(class_ids), np.nan)

    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "fpr": float(fpr_per_class[i]),
            "support": int(support[i]),
            "pr_auc": float(pr_auc_per_class[i]),
            "roc_auc": float(roc_auc_per_class[i]) if not np.isnan(roc_auc_per_class[i]) else None,
            "low_confidence": name in LOW_CONFIDENCE_CLASSES,
        }
        for i, name in enumerate(class_names)
    }

    stable_ids = [i for i, name in enumerate(class_names) if name not in LOW_CONFIDENCE_CLASSES]

    benign_id = label_map.get("BENIGN")

    return {
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": class_names,
        "per_class": per_class,
        "macro_recall_all_classes": float(recall.mean()),
        "macro_recall_stable_classes": float(recall[stable_ids].mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=class_ids, average="weighted", zero_division=0)),
        "macro_pr_auc": float(np.mean(pr_auc_per_class)),
        "macro_roc_auc": float(np.nanmean(roc_auc_per_class)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "aggregate_benign_fpr": aggregate_benign_fpr(y_true, y_pred, benign_id) if benign_id is not None else None,
    }
