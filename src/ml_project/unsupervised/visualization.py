"""Visualization functions for Mean Shift Clustering EDA and cluster analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from ml_project.core.visualization import save_correlation_heatmap, save_figure

CLUSTERING_PLOT_FILES: tuple[str, ...] = (
    "correlation_matrix.png",
    "clusters_pca.png",
    "cluster_size_distribution.png",
    "cluster_profile_heatmap.png",
    "mean_quality_by_cluster.png",
    "quality_by_cluster.png",
)


def save_clustering_eda(features: pd.DataFrame, reference: pd.Series, output_dir: Path) -> None:
    """Save descriptive statistics and correlation heatmap for original features."""
    eda_frame = features.copy()
    eda_frame[reference.name] = reference
    eda_frame.describe().T.to_csv(output_dir / "eda_summary.csv")
    save_correlation_heatmap(eda_frame, output_dir / "correlation_matrix.png")


def save_cluster_visualizations(
    scaled_features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    reference: pd.Series,
    output_dir: Path,
) -> None:
    """Save 2D PCA cluster scatter plot and quality-by-cluster boxplot."""
    pca = PCA(n_components=min(2, scaled_features.shape[1]))
    reduced_features = pca.fit_transform(scaled_features)
    reduced_centers = pca.transform(centers)

    if reduced_features.shape[1] >= 2:
        plot_frame = pd.DataFrame(
            {
                "PCA 1": reduced_features[:, 0],
                "PCA 2": reduced_features[:, 1],
                "cluster": labels.astype(str),
            }
        )
        figure, axis = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            data=plot_frame,
            x="PCA 1",
            y="PCA 2",
            hue="cluster",
            palette="tab20",
            alpha=0.7,
            s=35,
            ax=axis,
        )
        axis.scatter(
            reduced_centers[:, 0],
            reduced_centers[:, 1],
            marker="X",
            s=180,
            color="black",
            label="Pusat cluster",
            zorder=3,
        )
        explained = pca.explained_variance_ratio_.sum() * 100
        axis.set_title(f"Visualisasi Mean Shift dengan PCA (varian: {explained:.1f}%)")
        save_figure(figure, output_dir / "clusters_pca.png")

    quality_frame = pd.DataFrame({"cluster": labels.astype(str), reference.name: reference})
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=quality_frame, x="cluster", y=reference.name, ax=axis)
    axis.set_title("Distribusi Quality per Cluster (bukan fitur clustering)")
    axis.set(xlabel="Cluster", ylabel="Quality")
    save_figure(figure, output_dir / "quality_by_cluster.png")


def save_cluster_summary_plots(
    summary: pd.DataFrame,
    centers: np.ndarray,
    feature_names: pd.Index,
    reference_name: str,
    output_dir: Path,
) -> None:
    """Save cluster size, mean quality, and standardized cluster profile heatmap."""
    plot_summary = summary.sort_values("cluster")
    labels = plot_summary["cluster"].astype(str)

    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(labels, plot_summary["sample_count"], color="#4C78A8")
    axis.bar_label(bars, padding=3, fontsize=8)
    axis.set_yscale("log")
    axis.set(xlabel="Cluster", ylabel="Jumlah wine (skala log)")
    axis.set_title("Ukuran Setiap Cluster")
    save_figure(figure, output_dir / "cluster_size_distribution.png")

    mean_quality_col = f"mean_{reference_name}"
    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(labels, plot_summary[mean_quality_col], color="#59A14F")
    axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axis.set(xlabel="Cluster", ylabel=f"Rata-rata {reference_name}")
    axis.set_title("Rata-rata Quality per Cluster")
    save_figure(figure, output_dir / "mean_quality_by_cluster.png")

    center_frame = pd.DataFrame(
        centers,
        columns=feature_names,
        index=[f"Cluster {index}" for index in range(len(centers))],
    )
    figure, axis = plt.subplots(figsize=(15, max(5, len(center_frame) * 0.45)))
    sns.heatmap(center_frame, cmap="coolwarm", center=0, annot=False, ax=axis)
    axis.set_title("Profil Pusat Cluster dalam Skala Terstandarisasi")
    axis.set(xlabel="Fitur", ylabel="Cluster")
    save_figure(figure, output_dir / "cluster_profile_heatmap.png")
