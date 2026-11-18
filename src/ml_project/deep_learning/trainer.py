"""Deep learning training loop, checkpointing, and final validation export."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_project.core.io import save_json
from ml_project.core.paths import create_run_dir, resolve_image_dataset_dir
from ml_project.core.reproducibility import resolve_device, set_seed
from ml_project.core.visualization import open_saved_plots
from ml_project.deep_learning.config import TrainConfig
from ml_project.deep_learning.data import create_dataloaders
from ml_project.deep_learning.evaluator import evaluate_model, load_checkpoint
from ml_project.deep_learning.metrics import compute_classification_metrics
from ml_project.deep_learning.models.registry import create_model, get_architecture_spec
from ml_project.deep_learning.visualization import (
    DL_PLOT_FILES,
    save_confusion_matrix_plot,
    save_training_curves,
)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Execute one training epoch and return (average_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = correct / max(total_samples, 1)
    return avg_loss, accuracy


def _build_checkpoint_payload(
    model: nn.Module,
    *,
    model_name: str,
    class_names: list[str],
    image_size: int,
    weights: str,
    epoch: int,
    best_val_accuracy: float,
) -> dict[str, Any]:
    """Build a portable checkpoint dictionary using only tensors and built-in types."""
    return {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "num_classes": len(class_names),
        "class_names": class_names,
        "image_size": image_size,
        "weights": weights,
        "epoch": epoch,
        "best_val_accuracy": best_val_accuracy,
    }


def _save_run_config(
    config: TrainConfig,
    *,
    run_dir: Path,
    dataset_dir: Path,
    class_names: list[str],
    image_size: int,
    resize_size: int,
    device: torch.device,
) -> None:
    """Persist resolved experiment configuration as JSON."""
    payload = {
        "model": config.model,
        "weights": config.weights,
        "freeze_backbone": config.freeze_backbone,
        "dataset": str(dataset_dir),
        "class_names": class_names,
        "num_classes": len(class_names),
        "image_size": image_size,
        "resize_size": resize_size,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "num_workers": config.num_workers,
        "seed": config.seed,
        "device": str(device),
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    save_json(payload, run_dir / "config.json")


def run_training(config: TrainConfig) -> dict[str, Any]:
    """Train a registered vision model, track validation metrics, and export artifacts."""
    set_seed(config.seed)
    device = resolve_device(config.device)
    spec = get_architecture_spec(config.model)

    dataset_dir = resolve_image_dataset_dir(config.data_dir)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_label = config.run_name or f"{spec.name}_{timestamp}"
    run_dir = create_run_dir(config.output_dir, run_name=run_label)

    data_bundle = create_dataloaders(
        dataset_dir,
        spec,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(
        model_name=spec.name,
        num_classes=len(data_bundle.class_names),
        weights=config.weights,
        freeze_backbone=config.freeze_backbone,
    ).to(device)

    _save_run_config(
        config,
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        class_names=data_bundle.class_names,
        image_size=data_bundle.image_size,
        resize_size=data_bundle.resize_size,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    history_rows: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    best_checkpoint_path = run_dir / "best_model.pth"
    last_checkpoint_path = run_dir / "last_model.pth"

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            data_bundle.train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=config.epochs,
        )
        val_result = evaluate_model(
            model,
            data_bundle.val_loader,
            criterion,
            device,
            description=f"Epoch {epoch}/{config.epochs} [Val]",
        )
        val_metrics, _ = compute_classification_metrics(
            val_result.y_true,
            val_result.y_pred,
            data_bundle.class_names,
        )
        val_acc = float(val_metrics["accuracy"])
        val_macro_f1 = float(val_metrics["macro_f1"])

        scheduler.step()

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_result.loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            torch.save(
                _build_checkpoint_payload(
                    model,
                    model_name=spec.name,
                    class_names=data_bundle.class_names,
                    image_size=data_bundle.image_size,
                    weights=config.weights,
                    epoch=epoch,
                    best_val_accuracy=best_val_accuracy,
                ),
                best_checkpoint_path,
            )

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_result.loss:.4f} val_acc={val_acc:.4f} val_f1={val_macro_f1:.4f}"
        )

    torch.save(
        _build_checkpoint_payload(
            model,
            model_name=spec.name,
            class_names=data_bundle.class_names,
            image_size=data_bundle.image_size,
            weights=config.weights,
            epoch=config.epochs,
            best_val_accuracy=best_val_accuracy,
        ),
        last_checkpoint_path,
    )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(run_dir / "training_history.csv", index=False)
    save_training_curves(history_df, run_dir)

    best_checkpoint = load_checkpoint(best_checkpoint_path, device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_eval = evaluate_model(
        model,
        data_bundle.val_loader,
        criterion,
        device,
        description="Final Validation (Best Checkpoint)",
    )
    final_metrics, cm = compute_classification_metrics(
        final_eval.y_true,
        final_eval.y_pred,
        data_bundle.class_names,
    )
    save_confusion_matrix_plot(
        cm,
        data_bundle.class_names,
        run_dir / "confusion_matrix.png",
        title=f"Confusion Matrix — {spec.name} (Best Epoch {best_epoch})",
    )

    summary_metrics: dict[str, Any] = {
        "model": spec.name,
        "weights": config.weights,
        "freeze_backbone": config.freeze_backbone,
        "dataset": str(dataset_dir),
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "val_loss": final_eval.loss,
        "image_size": data_bundle.image_size,
        "resize_size": data_bundle.resize_size,
        "device": str(device),
        **final_metrics,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    save_json(summary_metrics, run_dir / "metrics.json")

    if config.show_plots:
        open_saved_plots(run_dir, DL_PLOT_FILES)

    return summary_metrics
