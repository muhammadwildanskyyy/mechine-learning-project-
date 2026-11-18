"""Tests for unified CLI argument parsing."""

from __future__ import annotations

from ml_project.cli import build_parser


def test_cli_dl_train_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["dl", "train"])
    assert args.command == "dl"
    assert args.dl_command == "train"
    assert args.model == "densenet169"
    assert args.weights == "imagenet"
    assert args.image_size is None


def test_cli_dl_train_efficientnet_b1() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["dl", "train", "--model", "efficientnet_b1", "--freeze-backbone", "--epochs", "5"]
    )
    assert args.model == "efficientnet_b1"
    assert args.weights == "imagenet"
    assert args.freeze_backbone is True
    assert args.epochs == 5


def test_cli_tabular_commands() -> None:
    parser = build_parser()
    reg_args = parser.parse_args(["linear-regression", "--no-show-plots"])
    assert reg_args.command == "linear-regression"
    assert reg_args.show_plots is False

    ms_args = parser.parse_args(["mean-shift", "--bandwidth", "1.2", "--no-show-plots"])
    assert ms_args.command == "mean-shift"
    assert ms_args.bandwidth == 1.2
    assert ms_args.show_plots is False
