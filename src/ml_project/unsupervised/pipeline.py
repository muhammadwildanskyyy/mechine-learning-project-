"""Mean Shift Clustering pipeline, evaluation, and artifact export."""

from __future__ import annotations

from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ml_project.core.io import load_wine_dataset, save_json
from ml_project.core.paths import resolve_wine_dataset_path
from ml_project.core.visualization import open_saved_plots
from ml_project.unsupervised.config import MeanShiftConfig
from ml_project.unsupervised.visualization import (
    CLUSTERING_PLOT_FILES,
    save_cluster_summary_plots,
    save_cluster_visualizations,
    save_clustering_eda,
)


def resolve_bandwidth(scaled_features: np.ndarray, config: MeanShiftConfig) -> float:
    """Use the explicit bandwidth or estimate it from standardized features."""
    if config.bandwidth is not None:
        return float(config.bandwidth)

    bandwidth = float(
        estimate_bandwidth(
            scaled_features,
            quantile=config.quantile,
            n_samples=min(config.bandwidth_samples, len(scaled_features)),
            random_state=config.random_state,
            n_jobs=-1,
        )
    )
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError(
            "Estimasi bandwidth menghasilkan nilai tidak valid. "
            "Tentukan nilai positif dengan --bandwidth."
        )
    return bandwidth


def build_cluster_summary(
    features: pd.DataFrame,
    reference: pd.Series,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Aggregate sample counts and feature means per cluster in original units."""
    clustered = features.copy()
    clustered[reference.name] = reference
    clustered["cluster"] = labels

    aggregation: dict[str, tuple[str, str]] = {
        "sample_count": ("cluster", "size"),
        f"mean_{reference.name}": (reference.name, "mean"),
    }
    aggregation.update({f"mean_{column}": (column, "mean") for column in features.columns})
    summary = clustered.groupby("cluster", sort=True).agg(**aggregation).reset_index()
    summary["percentage"] = summary["sample_count"] / len(clustered) * 100
    return summary.sort_values("sample_count", ascending=False)


def compute_clustering_scores(
    scaled_features: np.ndarray,
    labels: np.ndarray,
) -> tuple[float | None, float | None, str]:
    """Compute Silhouette and Davies-Bouldin scores when cluster count permits."""
    _, label_counts = np.unique(labels, return_counts=True)
    cluster_count = int(len(label_counts))
    singleton_count = int(np.sum(label_counts == 1))

    if not (2 <= cluster_count < len(scaled_features)):
        return (
            None,
            None,
            "Silhouette Score dan Davies-Bouldin Index tidak dihitung karena "
            "jumlah atau ukuran cluster tidak memenuhi syarat.",
        )

    silhouette = float(silhouette_score(scaled_features, labels))
    davies_bouldin = float(davies_bouldin_score(scaled_features, labels))
    note = "Metrik dihitung pada fitur yang telah distandarisasi."
    if singleton_count:
        note += (
            f" Terdapat {singleton_count} cluster berisi satu data; "
            "interpretasikan metrik dengan hati-hati."
        )
    return silhouette, davies_bouldin, note


def run_mean_shift(config: MeanShiftConfig) -> dict[str, object]:
    """Run preprocessing, Mean Shift clustering, evaluation, and artifact export."""
    data_path = resolve_wine_dataset_path(config.data_path)
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features, reference = load_wine_dataset(data_path, config.target, drop_missing_target=False)
    save_clustering_eda(features, reference, output_dir)

    # Target reference is intentionally excluded from clustering features.
    imputer = SimpleImputer(strategy="median")
    imputed_features = imputer.fit_transform(features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(imputed_features)

    bandwidth = resolve_bandwidth(scaled_features, config)
    clusterer = MeanShift(
        bandwidth=bandwidth,
        bin_seeding=config.bin_seeding,
        cluster_all=True,
        n_jobs=-1,
    )
    labels = clusterer.fit_predict(scaled_features)
    label_values, label_counts = np.unique(labels, return_counts=True)
    singleton_count = int(np.sum(label_counts == 1))

    silhouette, davies_bouldin, evaluation_note = compute_clustering_scores(scaled_features, labels)
    summary = build_cluster_summary(features, reference, labels)
    summary.to_csv(output_dir / "cluster_summary.csv", index=False)

    clustered_rows = features.copy()
    clustered_rows[reference.name] = reference
    clustered_rows["cluster"] = labels
    clustered_rows.to_csv(output_dir / "clustered_wines.csv", index=False)

    original_centers = scaler.inverse_transform(clusterer.cluster_centers_)
    pd.DataFrame(original_centers, columns=features.columns).assign(cluster=label_values).to_csv(
        output_dir / "cluster_centers.csv",
        index=False,
    )

    save_cluster_visualizations(
        scaled_features,
        labels,
        clusterer.cluster_centers_,
        reference,
        output_dir,
    )
    save_cluster_summary_plots(
        summary,
        clusterer.cluster_centers_,
        features.columns,
        reference.name,
        output_dir,
    )

    metrics: dict[str, object] = {
        "dataset": str(data_path),
        "excluded_target": config.target,
        "feature_count": int(features.shape[1]),
        "sample_count": int(len(features)),
        "cluster_count": int(len(label_values)),
        "largest_cluster_samples": int(label_counts.max()),
        "largest_cluster_percentage": float(label_counts.max() / len(labels) * 100),
        "singleton_cluster_count": singleton_count,
        "bandwidth": bandwidth,
        "bandwidth_source": "manual" if config.bandwidth is not None else "estimated",
        "bin_seeding": config.bin_seeding,
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies_bouldin,
        "evaluation_note": evaluation_note,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    save_json(metrics, output_dir / "metrics.json")
    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "clusterer": clusterer,
            "feature_names": list(features.columns),
            "excluded_target": config.target,
        },
        output_dir / "mean_shift_pipeline.joblib",
    )

    if config.show_plots:
        open_saved_plots(output_dir, CLUSTERING_PLOT_FILES)

    return metrics
