"""End-to-end smoke tests for Deep Learning data, training, and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml_project.deep_learning import EvalConfig, TrainConfig, run_evaluation, run_training
from ml_project.deep_learning.data import resolve_image_sizes
from ml_project.deep_learning.models import get_architecture_spec


def _create_synthetic_image_dataset(root: Path, classes: tuple[str, ...] = ("cat", "dog")) -> Path:
    rng = np.random.default_rng(7)
    for split, count in (("train", 4), ("val", 2)):
        for class_idx, class_name in enumerate(classes):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(count):
                base_color = 40 + class_idx * 150
                pixels = rng.integers(
                    low=max(0, base_color - 20),
                    high=min(255, base_color + 20),
                    size=(96, 96, 3),
                    dtype=np.uint8,
                )
                Image.fromarray(pixels, mode="RGB").save(class_dir / f"img_{idx}.png")
    return root


def test_resolve_image_sizes_matches_architecture_defaults() -> None:
    assert resolve_image_sizes(get_architecture_spec("densenet169"), None) == (224, 256)
    assert resolve_image_sizes(get_architecture_spec("efficientnet_b1"), None) == (240, 256)
    assert resolve_image_sizes(get_architecture_spec("densenet169"), 160) == (160, 183)


@pytest.mark.parametrize(
    ("model_name", "expected_size"),
    [
        ("densenet169", 224),
        ("efficientnet_b1", 240),
    ],
)
def test_dl_train_and_evaluate_end_to_end(
    tmp_path: Path,
    model_name: str,
    expected_size: int,
) -> None:
    dataset_dir = _create_synthetic_image_dataset(tmp_path / "imagenette_mock")
    outputs_base = tmp_path / "outputs"

    train_metrics = run_training(
        TrainConfig(
            model=model_name,
            weights="none",
            data_dir=dataset_dir,
            output_dir=outputs_base,
            run_name=f"test_{model_name}",
            epochs=1,
            batch_size=2,
            num_workers=0,
            device="cpu",
            freeze_backbone=True,
            show_plots=False,
        )
    )

    run_dir = outputs_base / f"test_{model_name}"
    assert train_metrics["model"] == model_name
    assert train_metrics["image_size"] == expected_size
    assert (run_dir / "best_model.pth").is_file()
    assert (run_dir / "last_model.pth").is_file()
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "training_history.csv").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "loss_curve.png").is_file()
    assert (run_dir / "accuracy_curve.png").is_file()
    assert (run_dir / "confusion_matrix.png").is_file()

    eval_metrics = run_evaluation(
        EvalConfig(
            checkpoint=run_dir / "best_model.pth",
            data_dir=dataset_dir,
            batch_size=2,
            num_workers=0,
            device="cpu",
            show_plots=False,
        )
    )

    assert eval_metrics["model"] == model_name
    assert eval_metrics["image_size"] == expected_size
    assert (run_dir / "eval_val_metrics.json").is_file()
    assert (run_dir / "eval_val_confusion_matrix.png").is_file()
