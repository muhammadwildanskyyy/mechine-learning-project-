"""Unit tests for the Deep Learning model registry and factory."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from ml_project.deep_learning.models import (
    create_model,
    get_architecture_spec,
    list_architectures,
)


def test_registered_architectures_and_default_sizes() -> None:
    architectures = list_architectures()
    assert "densenet169" in architectures
    assert "efficientnet_b1" in architectures

    densenet_spec = get_architecture_spec("densenet169")
    efficientnet_spec = get_architecture_spec("efficientnet_b1")
    assert densenet_spec.default_image_size == 224
    assert efficientnet_spec.default_image_size == 240


@pytest.mark.parametrize(
    ("model_name", "image_size"),
    [
        ("densenet169", 224),
        ("efficientnet_b1", 240),
    ],
)
def test_create_model_forward_shape_and_freeze(model_name: str, image_size: int) -> None:
    model = create_model(
        model_name=model_name,
        num_classes=10,
        weights="none",
        freeze_backbone=True,
    )
    model.eval()
    inputs = torch.randn(2, 3, image_size, image_size)
    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (2, 10)

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    frozen_names = [name for name, param in model.named_parameters() if not param.requires_grad]
    assert trainable_names, "Classification head parameters must remain trainable."
    assert frozen_names, "Backbone parameters must be frozen when freeze_backbone=True."
    assert all(name.startswith("classifier.") for name in trainable_names)


def test_create_model_passes_pretrained_flag() -> None:
    with patch("ml_project.deep_learning.models.densenet.densenet169") as mock_densenet:
        fake_model = torch.nn.Module()
        fake_model.classifier = torch.nn.Linear(1664, 1000)
        mock_densenet.return_value = fake_model

        create_model("densenet169", num_classes=5, weights="imagenet")
        _, kwargs = mock_densenet.call_args
        assert kwargs["weights"] is not None


def test_create_model_validation_errors() -> None:
    with pytest.raises(ValueError, match="Arsitektur"):
        create_model("unknown_net", num_classes=10, weights="none")

    with pytest.raises(ValueError, match="Opsi weights"):
        create_model("densenet169", num_classes=10, weights="invalid")

    with pytest.raises(ValueError, match="num_classes"):
        create_model("densenet169", num_classes=0, weights="none")
