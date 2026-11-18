"""Configuration dataclasses for Deep Learning training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ml_project.core.paths import DEFAULT_OUTPUTS_DIR
from ml_project.deep_learning.models.registry import SUPPORTED_WEIGHTS, get_architecture_spec


@dataclass(slots=True)
class TrainConfig:
    """Parameters governing a Deep Learning training run."""

    model: str = "densenet169"
    weights: str = "imagenet"
    data_dir: Path | None = None
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUTS_DIR / "deep_learning")
    run_name: str | None = None
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    image_size: int | None = None
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"
    freeze_backbone: bool = False
    show_plots: bool = False

    def __post_init__(self) -> None:
        get_architecture_spec(self.model)
        if self.weights.strip().lower() not in SUPPORTED_WEIGHTS:
            allowed = ", ".join(SUPPORTED_WEIGHTS)
            raise ValueError(f"--weights harus salah satu dari: {allowed}.")
        if self.epochs < 1:
            raise ValueError("--epochs harus minimal 1.")
        if self.batch_size < 1:
            raise ValueError("--batch-size harus minimal 1.")
        if self.lr <= 0:
            raise ValueError("--lr harus lebih besar dari 0.")
        if self.weight_decay < 0:
            raise ValueError("--weight-decay tidak boleh negatif.")
        if self.image_size is not None and self.image_size < 32:
            raise ValueError("--image-size harus minimal 32.")
        if self.num_workers < 0:
            raise ValueError("--num-workers tidak boleh negatif.")


@dataclass(slots=True)
class EvalConfig:
    """Parameters governing standalone checkpoint evaluation."""

    checkpoint: Path
    data_dir: Path | None = None
    split: str = "val"
    batch_size: int = 32
    image_size: int | None = None
    num_workers: int = 4
    device: str = "auto"
    output_dir: Path | None = None
    show_plots: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("--batch-size harus minimal 1.")
        if self.image_size is not None and self.image_size < 32:
            raise ValueError("--image-size harus minimal 32.")
        if self.num_workers < 0:
            raise ValueError("--num-workers tidak boleh negatif.")
