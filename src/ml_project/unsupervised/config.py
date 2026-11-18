"""Configuration dataclass for Mean Shift Clustering experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ml_project.core.paths import DEFAULT_OUTPUTS_DIR


@dataclass(slots=True)
class MeanShiftConfig:
    """Parameters governing a Wine Quality Mean Shift Clustering run."""

    data_path: Path | None = None
    target: str = "quality"
    bandwidth: float | None = None
    quantile: float = 0.2
    bandwidth_samples: int = 500
    random_state: int = 42
    bin_seeding: bool = True
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUTS_DIR / "mean_shift")
    show_plots: bool = True

    def __post_init__(self) -> None:
        if self.bandwidth is not None and self.bandwidth <= 0:
            raise ValueError("--bandwidth harus bernilai lebih besar dari 0.")
        if not 0 < self.quantile <= 1:
            raise ValueError("--quantile harus lebih besar dari 0 dan paling besar 1.")
        if self.bandwidth_samples < 1:
            raise ValueError("--bandwidth-samples harus minimal 1.")
