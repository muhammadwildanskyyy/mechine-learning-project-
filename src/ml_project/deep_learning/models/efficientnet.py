"""EfficientNet architecture builders and specifications."""

from __future__ import annotations

from torch import nn
from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1

from ml_project.deep_learning.models.base import ArchitectureSpec


def build_efficientnet_b1(num_classes: int, pretrained: bool) -> nn.Module:
    """Construct EfficientNet-B1 with a classifier head matching num_classes."""
    weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
    model = efficientnet_b1(weights=weights)
    last_layer = model.classifier[-1]
    if not isinstance(last_layer, nn.Linear):
        raise TypeError("Struktur classifier EfficientNet-B1 tidak sesuai ekspektasi.")
    model.classifier[-1] = nn.Linear(last_layer.in_features, num_classes)
    return model


EFFICIENTNET_B1_SPEC = ArchitectureSpec(
    name="efficientnet_b1",
    family="efficientnet",
    default_image_size=240,
    default_resize_size=256,
    builder=build_efficientnet_b1,
    head_attr="classifier",
)
