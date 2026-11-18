"""Shared model evaluation loop and standalone checkpoint evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_project.core.io import save_json
from ml_project.core.paths import resolve_image_dataset_dir
from ml_project.core.reproducibility import resolve_device
from ml_project.core.visualization import open_saved_plots
from ml_project.deep_learning.config import EvalConfig
from ml_project.deep_learning.data import create_eval_dataloader
from ml_project.deep_learning.metrics import compute_classification_metrics
from ml_project.deep_learning.models.registry import create_model, get_architecture_spec
from ml_project.deep_learning.visualization import save_confusion_matrix_plot


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Predictions and aggregate loss from evaluating a model over a DataLoader."""

    loss: float
    y_true: list[int]
    y_pred: list[int]


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    description: str = "Evaluating",
) -> EvaluationResult:
    """Run forward pass over a DataLoader without gradient tracking."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, targets in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        predictions = logits.argmax(dim=1)
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    return EvaluationResult(loss=avg_loss, y_true=y_true, y_pred=y_pred)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    """Load a saved training checkpoint dictionary."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Format checkpoint tidak valid: {checkpoint_path}")
    return checkpoint


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    """Load a checkpoint and evaluate it on the target ImageFolder split."""
    device = resolve_device(config.device)
    checkpoint = load_checkpoint(config.checkpoint.resolve(), device)

    model_name = str(checkpoint["model_name"])
    num_classes = int(checkpoint["num_classes"])
    checkpoint_classes = list(checkpoint["class_names"])
    checkpoint_image_size = checkpoint.get("image_size")
    resolved_image_size = (
        config.image_size if config.image_size is not None else checkpoint_image_size
    )

    spec = get_architecture_spec(model_name)
    dataset_root = resolve_image_dataset_dir(config.data_dir)
    split_dir = dataset_root / config.split

    loader, split_classes, crop_size, resize_size = create_eval_dataloader(
        split_dir=split_dir,
        spec=spec,
        batch_size=config.batch_size,
        image_size=int(resolved_image_size) if resolved_image_size is not None else None,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    if split_classes != checkpoint_classes:
        raise ValueError(
            "Daftar kelas pada dataset evaluasi tidak cocok dengan checkpoint: "
            f"dataset={split_classes}, checkpoint={checkpoint_classes}."
        )

    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        weights="none",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    eval_result = evaluate_model(
        model,
        loader,
        criterion,
        device,
        description=f"Eval ({config.split})",
    )
    classification_metrics, cm = compute_classification_metrics(
        eval_result.y_true,
        eval_result.y_pred,
        checkpoint_classes,
    )

    output_dir = (
        config.output_dir.resolve()
        if config.output_dir is not None
        else config.checkpoint.resolve().parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_filename = f"eval_{config.split}_confusion_matrix.png"
    save_confusion_matrix_plot(
        cm,
        checkpoint_classes,
        output_dir / cm_filename,
        title=f"Confusion Matrix — {model_name} ({config.split})",
    )

    metrics: dict[str, Any] = {
        "checkpoint": str(config.checkpoint.resolve()),
        "dataset": str(dataset_root),
        "split": config.split,
        "model": model_name,
        "image_size": crop_size,
        "resize_size": resize_size,
        "device": str(device),
        "loss": eval_result.loss,
        **classification_metrics,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
    }
    save_json(metrics, output_dir / f"eval_{config.split}_metrics.json")

    if config.show_plots:
        open_saved_plots(output_dir, (cm_filename,))

    return metrics
