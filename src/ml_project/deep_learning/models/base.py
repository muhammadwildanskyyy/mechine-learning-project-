"""Base specification types for pluggable classification architectures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

ModelBuilder = Callable[[int, bool], nn.Module]


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    """Metadata and builder definition for a registered vision backbone."""

    name: str
    family: str
    default_image_size: int
    default_resize_size: int
    builder: ModelBuilder
    head_attr: str = "classifier"
