"""Image dataset loading and architecture-standard preprocessing transforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml_project.deep_learning.models.base import ArchitectureSpec

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class DataBundle:
    """Container for train/val loaders and resolved class metadata."""

    train_loader: DataLoader
    val_loader: DataLoader
    class_names: list[str]
    image_size: int
    resize_size: int


def resolve_image_sizes(spec: ArchitectureSpec, requested_size: int | None) -> tuple[int, int]:
    """Return (crop_size, resize_size) aligned with architecture defaults."""
    if requested_size is None:
        return spec.default_image_size, spec.default_resize_size

    if requested_size < 32:
        raise ValueError("image_size harus minimal 32.")
    scaled_resize = max(requested_size, int(round(requested_size * 256 / 224)))
    return requested_size, scaled_resize


def build_transforms(image_size: int, resize_size: int, *, is_train: bool) -> transforms.Compose:
    """Construct standard ImageNet training or evaluation image transforms."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def create_dataloaders(
    data_dir: Path,
    spec: ArchitectureSpec,
    *,
    batch_size: int,
    image_size: int | None = None,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> DataBundle:
    """Create training and validation DataLoaders from an ImageFolder directory."""
    resolved_crop, resolved_resize = resolve_image_sizes(spec, image_size)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise ValueError(f"Direktori '{data_dir}' harus memiliki subfolder 'train/' dan 'val/'.")

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=build_transforms(resolved_crop, resolved_resize, is_train=True),
    )
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=build_transforms(resolved_crop, resolved_resize, is_train=False),
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Dataset train/ atau val/ tidak boleh kosong.")
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Daftar kelas pada subfolder train/ dan val/ tidak konsisten.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=list(train_dataset.classes),
        image_size=resolved_crop,
        resize_size=resolved_resize,
    )


def create_eval_dataloader(
    split_dir: Path,
    spec: ArchitectureSpec,
    *,
    batch_size: int,
    image_size: int | None = None,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> tuple[DataLoader, list[str], int, int]:
    """Create an evaluation DataLoader for a single ImageFolder split directory."""
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Direktori split evaluasi tidak ditemukan: {split_dir}")

    resolved_crop, resolved_resize = resolve_image_sizes(spec, image_size)
    dataset = datasets.ImageFolder(
        root=split_dir,
        transform=build_transforms(resolved_crop, resolved_resize, is_train=False),
    )
    if len(dataset) == 0:
        raise ValueError(f"Direktori evaluasi '{split_dir}' tidak memiliki gambar.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, list(dataset.classes), resolved_crop, resolved_resize
