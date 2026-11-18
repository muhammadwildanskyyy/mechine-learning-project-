"""Classification evaluation metrics for multi-class image tasks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> tuple[dict[str, object], np.ndarray]:
    """Compute accuracy, macro precision/recall/F1, per-class report, and confusion matrix."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = list(range(len(class_names)))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "sample_count": int(len(y_true_arr)),
        "per_class_report": report,
    }
    return metrics, cm
