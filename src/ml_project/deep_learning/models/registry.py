"""Architecture registry and model factory for deep learning classification."""

from __future__ import annotations

from torch import nn

from ml_project.deep_learning.models.base import ArchitectureSpec
from ml_project.deep_learning.models.densenet import DENSENET169_SPEC
from ml_project.deep_learning.models.efficientnet import EFFICIENTNET_B1_SPEC

SUPPORTED_WEIGHTS: tuple[str, ...] = ("imagenet", "none")

_REGISTRY: dict[str, ArchitectureSpec] = {}


def register_architecture(spec: ArchitectureSpec) -> None:
    """Register an architecture specification under its canonical lowercase name."""
    key = spec.name.strip().lower()
    _REGISTRY[key] = spec


for _spec in (DENSENET169_SPEC, EFFICIENTNET_B1_SPEC):
    register_architecture(_spec)


def list_architectures() -> tuple[str, ...]:
    """Return all registered architecture names."""
    return tuple(_REGISTRY.keys())


def get_architecture_spec(model_name: str) -> ArchitectureSpec:
    """Look up an ArchitectureSpec by name or raise a descriptive ValueError."""
    normalized = model_name.strip().lower()
    if normalized not in _REGISTRY:
        available = ", ".join(list_architectures())
        raise ValueError(
            f"Arsitektur '{model_name}' tidak tersedia. Pilihan yang didukung: {available}."
        )
    return _REGISTRY[normalized]


def freeze_model_backbone(model: nn.Module, head_attr: str) -> None:
    """Freeze all parameters except the classification head."""
    head = getattr(model, head_attr, None)
    if not isinstance(head, nn.Module):
        raise AttributeError(
            f"Model '{type(model).__name__}' tidak memiliki atribut head '{head_attr}'."
        )

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head.parameters():
        parameter.requires_grad = True


def create_model(
    model_name: str,
    num_classes: int = 10,
    weights: str = "imagenet",
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a classification model from registry parameters."""
    if num_classes < 1:
        raise ValueError("num_classes harus minimal 1.")

    normalized_weights = weights.strip().lower()
    if normalized_weights not in SUPPORTED_WEIGHTS:
        allowed = ", ".join(SUPPORTED_WEIGHTS)
        raise ValueError(f"Opsi weights '{weights}' tidak valid. Pilihan yang didukung: {allowed}.")

    spec = get_architecture_spec(model_name)
    pretrained = normalized_weights == "imagenet"
    model = spec.builder(num_classes, pretrained)

    if freeze_backbone:
        freeze_model_backbone(model, spec.head_attr)

    return model
