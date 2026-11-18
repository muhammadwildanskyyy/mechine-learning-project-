"""DenseNet architecture builders and specifications."""

from __future__ import annotations

from torch import nn
from torchvision.models import DenseNet169_Weights, densenet169

from ml_project.deep_learning.models.base import ArchitectureSpec


def build_densenet169(num_classes: int, pretrained: bool) -> nn.Module:
    """Construct DenseNet-169 with a classifier head matching num_classes."""
    weights = DenseNet169_Weights.DEFAULT if pretrained else None
    model = densenet169(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


DENSENET169_SPEC = ArchitectureSpec(
    name="densenet169",
    family="densenet",
    default_image_size=224,
    default_resize_size=256,
    builder=build_densenet169,
    head_attr="classifier",
)
