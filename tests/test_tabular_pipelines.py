"""Tests for Linear Regression and Mean Shift Clustering pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_project.supervised import LinearRegressionConfig, run_linear_regression
from ml_project.unsupervised import MeanShiftConfig, run_mean_shift


def _write_synthetic_wine_csv(csv_path: Path, rows: int = 60) -> Path:
    rng = np.random.default_rng(42)
    alcohol = rng.uniform(8.5, 14.0, size=rows)
    acidity = rng.uniform(0.2, 1.0, size=rows)
    sugar = rng.uniform(1.0, 15.0, size=rows)
    quality = np.clip(np.round(4.0 + 0.35 * alcohol - 1.2 * acidity), 3, 9)

    frame = pd.DataFrame(
        {
            "fixed acidity": rng.uniform(5.0, 10.0, size=rows),
            "volatile acidity": acidity,
            "residual sugar": sugar,
            "alcohol": alcohol,
            "quality": quality,
        }
    )
    frame.to_csv(csv_path, sep=";", index=False)
    return csv_path


def test_linear_regression_pipeline_exports_artifacts(tmp_path: Path) -> None:
    csv_path = _write_synthetic_wine_csv(tmp_path / "wine.csv")
    output_dir = tmp_path / "out_reg"

    metrics = run_linear_regression(
        LinearRegressionConfig(
            data_path=csv_path,
            output_dir=output_dir,
            show_plots=False,
        )
    )

    assert metrics["feature_count"] == 4
    assert metrics["train_samples"] + metrics["test_samples"] == 60
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "coefficients.csv").is_file()
    assert (output_dir / "test_predictions.csv").is_file()
    assert (output_dir / "linear_regression_pipeline.joblib").is_file()
    assert (output_dir / "actual_vs_predicted.png").is_file()
    assert (output_dir / "feature_importance.png").is_file()

    saved_metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics["r2"] == pytest.approx(metrics["r2"])


def test_mean_shift_pipeline_excludes_target_and_exports_artifacts(tmp_path: Path) -> None:
    csv_path = _write_synthetic_wine_csv(tmp_path / "wine.csv")
    output_dir = tmp_path / "out_cluster"

    metrics = run_mean_shift(
        MeanShiftConfig(
            data_path=csv_path,
            bandwidth=1.5,
            output_dir=output_dir,
            show_plots=False,
        )
    )

    assert metrics["excluded_target"] == "quality"
    assert metrics["feature_count"] == 4
    assert metrics["bandwidth_source"] == "manual"
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "cluster_summary.csv").is_file()
    assert (output_dir / "clustered_wines.csv").is_file()
    assert (output_dir / "cluster_centers.csv").is_file()
    assert (output_dir / "mean_shift_pipeline.joblib").is_file()
    assert (output_dir / "clusters_pca.png").is_file()

    centers = pd.read_csv(output_dir / "cluster_centers.csv")
    assert "quality" not in centers.columns


def test_tabular_config_validation() -> None:
    with pytest.raises(ValueError, match="--test-size"):
        LinearRegressionConfig(test_size=1.2)

    with pytest.raises(ValueError, match="--bandwidth"):
        MeanShiftConfig(bandwidth=0.0)

    with pytest.raises(ValueError, match="--quantile"):
        MeanShiftConfig(quantile=1.5)
