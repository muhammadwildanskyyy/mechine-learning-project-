"""Linear Regression training, evaluation, and artifact export pipeline."""

from __future__ import annotations

from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_project.core.io import load_wine_dataset, save_json
from ml_project.core.paths import resolve_wine_dataset_path
from ml_project.core.visualization import open_saved_plots
from ml_project.supervised.config import LinearRegressionConfig
from ml_project.supervised.visualization import (
    REGRESSION_PLOT_FILES,
    save_feature_importance_plot,
    save_prediction_plots,
    save_regression_eda,
)


def build_regression_pipeline() -> Pipeline:
    """Build the leakage-free sklearn regression pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def run_linear_regression(config: LinearRegressionConfig) -> dict[str, float | int | str]:
    """Run EDA, train/test split, Linear Regression training, and artifact export."""
    data_path = resolve_wine_dataset_path(config.data_path)
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features, target = load_wine_dataset(data_path, config.target, drop_missing_target=True)
    save_regression_eda(features, target, output_dir)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    pipeline = build_regression_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    actual_array = y_test.to_numpy()
    mse = float(mean_squared_error(actual_array, predictions))
    metrics: dict[str, float | int | str] = {
        "dataset": str(data_path),
        "target": config.target,
        "random_state": config.random_state,
        "test_size": config.test_size,
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "feature_count": int(features.shape[1]),
        "mae": float(mean_absolute_error(actual_array, predictions)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual_array, predictions)),
        "mean_actual_quality": float(y_test.mean()),
        "mean_predicted_quality": float(np.mean(predictions)),
        "mean_residual": float(np.mean(actual_array - predictions)),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    pd.DataFrame(
        {
            "row_index": x_test.index,
            "actual_quality": actual_array,
            "predicted_quality": predictions,
            "residual": actual_array - predictions,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    regression: LinearRegression = pipeline.named_steps["model"]
    coefficients = pd.DataFrame(
        {
            "feature": features.columns,
            "standardized_coefficient": regression.coef_,
            "absolute_coefficient": np.abs(regression.coef_),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    coefficients.to_csv(output_dir / "coefficients.csv", index=False)

    save_json(metrics, output_dir / "metrics.json")
    joblib.dump(pipeline, output_dir / "linear_regression_pipeline.joblib")
    save_prediction_plots(actual_array, predictions, output_dir)
    save_feature_importance_plot(coefficients, output_dir)

    if config.show_plots:
        open_saved_plots(output_dir, REGRESSION_PLOT_FILES)

    return metrics
