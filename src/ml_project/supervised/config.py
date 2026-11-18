"""Configuration dataclass for Linear Regression experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ml_project.core.paths import DEFAULT_OUTPUTS_DIR


@dataclass(slots=True)
class LinearRegressionConfig:
    """Parameters governing a Wine Quality Linear Regression run."""

    data_path: Path | None = None
    target: str = "quality"
    test_size: float = 0.2
    random_state: int = 42
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUTS_DIR / "linear_regression")
    show_plots: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("--test-size harus lebih besar dari 0 dan kurang dari 1.")
