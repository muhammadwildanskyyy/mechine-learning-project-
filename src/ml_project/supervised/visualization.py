"""Visualization functions for Linear Regression EDA and evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml_project.core.visualization import save_correlation_heatmap, save_figure

REGRESSION_PLOT_FILES: tuple[str, ...] = (
    "target_distribution.png",
    "correlation_matrix.png",
    "actual_vs_predicted.png",
    "residual_plot.png",
    "residual_distribution.png",
    "feature_importance.png",
)


def save_regression_eda(features: pd.DataFrame, target: pd.Series, output_dir: Path) -> None:
    """Save statistical summary, correlation heatmap, and target distribution chart."""
    eda_frame = features.copy()
    eda_frame[target.name] = target
    eda_frame.describe().T.to_csv(output_dir / "eda_summary.csv")

    save_correlation_heatmap(eda_frame, output_dir / "correlation_matrix.png")

    target_counts = target.value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        target_counts.index.astype(str),
        target_counts.values,
        color="#4C78A8",
        width=0.65,
    )
    axis.bar_label(bars, padding=3, fontsize=9)
    axis.set(xlabel="Quality", ylabel="Jumlah wine")
    axis.set_title("Distribusi Target Quality")
    save_figure(figure, output_dir / "target_distribution.png")


def save_prediction_plots(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_dir: Path,
) -> None:
    """Save actual-vs-predicted scatter, residual scatter, and residual histogram."""
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))

    figure, axis = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=actual, y=predicted, alpha=0.65, ax=axis)
    axis.plot([lower, upper], [lower, upper], "r--", label="Prediksi sempurna")
    axis.set(xlabel="Quality aktual", ylabel="Quality prediksi")
    axis.set_title("Aktual vs Prediksi")
    axis.legend()
    save_figure(figure, output_dir / "actual_vs_predicted.png")

    residuals = actual - predicted
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=predicted, y=residuals, alpha=0.65, ax=axis)
    axis.axhline(0, color="red", linestyle="--")
    axis.set(xlabel="Quality prediksi", ylabel="Residual (aktual - prediksi)")
    axis.set_title("Residual terhadap Prediksi")
    save_figure(figure, output_dir / "residual_plot.png")

    figure, axis = plt.subplots(figsize=(7, 5))
    sns.histplot(residuals, bins=30, kde=True, color="#59A14F", ax=axis)
    axis.axvline(0, color="red", linestyle="--", label="Residual nol")
    axis.set(xlabel="Residual (aktual - prediksi)", ylabel="Frekuensi")
    axis.set_title("Distribusi Residual")
    axis.legend()
    save_figure(figure, output_dir / "residual_distribution.png")


def save_feature_importance_plot(coefficients: pd.DataFrame, output_dir: Path) -> None:
    """Save horizontal bar chart of standardized regression coefficients."""
    plot_data = coefficients.sort_values("standardized_coefficient")
    colors = np.where(plot_data["standardized_coefficient"] >= 0, "#4C78A8", "#E45756")

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(plot_data["feature"], plot_data["standardized_coefficient"], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Koefisien terstandarisasi", ylabel="Fitur")
    axis.set_title("Pengaruh Fitur pada Prediksi Quality")
    save_figure(figure, output_dir / "feature_importance.png")
