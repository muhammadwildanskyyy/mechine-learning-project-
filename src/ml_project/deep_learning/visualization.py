"""Visualization utilities for deep learning training curves and confusion matrices."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml_project.core.visualization import save_figure

DL_PLOT_FILES: tuple[str, ...] = (
    "loss_curve.png",
    "accuracy_curve.png",
    "confusion_matrix.png",
)


def save_training_curves(history: pd.DataFrame, output_dir: Path) -> None:
    """Save train/val loss and accuracy curve plots from epoch history."""
    epochs = history["epoch"]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axis.plot(epochs, history["val_loss"], marker="s", label="Val Loss")
    axis.set(xlabel="Epoch", ylabel="Loss")
    axis.set_title("Kurva Loss Pelatihan vs Validasi")
    axis.legend()
    axis.grid(True, linestyle="--", alpha=0.4)
    save_figure(figure, output_dir / "loss_curve.png")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, history["train_accuracy"], marker="o", label="Train Accuracy")
    axis.plot(epochs, history["val_accuracy"], marker="s", label="Val Accuracy")
    axis.set(xlabel="Epoch", ylabel="Accuracy")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Kurva Akurasi Pelatihan vs Validasi")
    axis.legend()
    axis.grid(True, linestyle="--", alpha=0.4)
    save_figure(figure, output_dir / "accuracy_curve.png")


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Render and save a labeled confusion matrix heatmap."""
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=axis,
    )
    axis.set(xlabel="Prediksi", ylabel="Aktual")
    axis.set_title(title)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    plt.setp(axis.get_yticklabels(), rotation=0)
    return save_figure(figure, output_path)
