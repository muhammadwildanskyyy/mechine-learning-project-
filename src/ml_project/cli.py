"""Unified command-line interface for all ML pipelines."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ml_project.core.paths import DEFAULT_OUTPUTS_DIR
from ml_project.deep_learning.config import EvalConfig, TrainConfig
from ml_project.deep_learning.evaluator import run_evaluation
from ml_project.deep_learning.models.registry import SUPPORTED_WEIGHTS, list_architectures
from ml_project.deep_learning.trainer import run_training
from ml_project.supervised.config import LinearRegressionConfig
from ml_project.supervised.pipeline import run_linear_regression
from ml_project.unsupervised.config import MeanShiftConfig
from ml_project.unsupervised.pipeline import run_mean_shift


def _add_linear_regression_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "linear-regression",
        help="Latih dan evaluasi Linear Regression pada dataset Wine Quality.",
    )
    parser.add_argument("--data-path", type=Path, help="Path CSV Wine Quality.")
    parser.add_argument(
        "--target",
        default="quality",
        help="Kolom target regresi (default: quality).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi data uji (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed split (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR / "linear_regression",
        help="Direktori output eksperimen.",
    )
    parser.add_argument(
        "--no-show-plots",
        dest="show_plots",
        action="store_false",
        help="Jangan buka grafik setelah selesai.",
    )
    parser.set_defaults(show_plots=True, handler=_handle_linear_regression)


def _add_mean_shift_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "mean-shift",
        help="Jalankan Mean Shift Clustering pada dataset Wine Quality.",
    )
    parser.add_argument("--data-path", type=Path, help="Path CSV Wine Quality.")
    parser.add_argument(
        "--target",
        default="quality",
        help="Kolom referensi yang dikecualikan dari fitur cluster (default: quality).",
    )
    parser.add_argument("--bandwidth", type=float, help="Bandwidth manual Mean Shift.")
    parser.add_argument("--quantile", type=float, default=0.2, help="Quantile estimasi bandwidth.")
    parser.add_argument(
        "--bandwidth-samples",
        type=int,
        default=500,
        help="Jumlah sampel maksimum untuk estimasi bandwidth.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Seed sampling bandwidth.")
    parser.add_argument(
        "--no-bin-seeding",
        dest="bin_seeding",
        action="store_false",
        help="Nonaktifkan bin seeding Mean Shift.",
    )
    parser.set_defaults(bin_seeding=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR / "mean_shift",
        help="Direktori output eksperimen.",
    )
    parser.add_argument(
        "--no-show-plots",
        dest="show_plots",
        action="store_false",
        help="Jangan buka grafik setelah selesai.",
    )
    parser.set_defaults(show_plots=True, handler=_handle_mean_shift)


def _add_deep_learning_parser(subparsers: argparse._SubParsersAction) -> None:
    dl_parser = subparsers.add_parser(
        "dl",
        help="Pipeline Deep Learning (training dan evaluasi klasifikasi gambar).",
    )
    dl_subparsers = dl_parser.add_subparsers(dest="dl_command", required=True)

    train_parser = dl_subparsers.add_parser(
        "train",
        help="Latih model klasifikasi gambar dengan arsitektur pilihan.",
    )
    train_parser.add_argument(
        "--model",
        default="densenet169",
        choices=list_architectures(),
        help="Arsitektur model (default: densenet169).",
    )
    train_parser.add_argument(
        "--weights",
        default="imagenet",
        choices=SUPPORTED_WEIGHTS,
        help="Sumber bobot awal (default: imagenet).",
    )
    train_parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Bekukan bobot backbone dan latih classifier head saja.",
    )
    train_parser.add_argument(
        "--data-dir",
        type=Path,
        help="Direktori dataset berisi train/ dan val/.",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR / "deep_learning",
        help="Direktori dasar penyimpanan run.",
    )
    train_parser.add_argument("--run-name", help="Nama folder run (opsional).")
    train_parser.add_argument("--epochs", type=int, default=10, help="Jumlah epoch (default: 10).")
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Ukuran batch (default: 32).",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate AdamW (default: 0.001).",
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay AdamW (default: 0.0001).",
    )
    train_parser.add_argument(
        "--image-size",
        type=int,
        help="Override ukuran input gambar; default mengikuti standar arsitektur.",
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Jumlah worker DataLoader.",
    )
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed pelatihan.")
    train_parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Compute device (default: auto).",
    )
    train_parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Buka grafik hasil pelatihan setelah selesai.",
    )
    train_parser.set_defaults(handler=_handle_dl_train)

    eval_parser = dl_subparsers.add_parser(
        "evaluate",
        help="Evaluasi checkpoint model Deep Learning yang sudah dilatih.",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path file checkpoint .pth.",
    )
    eval_parser.add_argument(
        "--data-dir",
        type=Path,
        help="Direktori dataset berisi train/ dan val/.",
    )
    eval_parser.add_argument(
        "--split",
        default="val",
        help="Subfolder split evaluasi (default: val).",
    )
    eval_parser.add_argument("--batch-size", type=int, default=32, help="Ukuran batch evaluasi.")
    eval_parser.add_argument(
        "--image-size",
        type=int,
        help="Override ukuran input gambar; default membaca dari checkpoint.",
    )
    eval_parser.add_argument("--num-workers", type=int, default=4, help="Jumlah worker DataLoader.")
    eval_parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Compute device (default: auto).",
    )
    eval_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Direktori output evaluasi (default: folder yang sama dengan checkpoint).",
    )
    eval_parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Buka grafik confusion matrix setelah evaluasi.",
    )
    eval_parser.set_defaults(handler=_handle_dl_eval)


def _handle_linear_regression(args: argparse.Namespace) -> dict[str, object]:
    config = LinearRegressionConfig(
        data_path=args.data_path,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
    )
    metrics = run_linear_regression(config)
    print("Eksperimen Linear Regression selesai.")
    return dict(metrics)


def _handle_mean_shift(args: argparse.Namespace) -> dict[str, object]:
    config = MeanShiftConfig(
        data_path=args.data_path,
        target=args.target,
        bandwidth=args.bandwidth,
        quantile=args.quantile,
        bandwidth_samples=args.bandwidth_samples,
        random_state=args.random_state,
        bin_seeding=args.bin_seeding,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
    )
    metrics = run_mean_shift(config)
    print("Eksperimen Mean Shift selesai.")
    return metrics


def _handle_dl_train(args: argparse.Namespace) -> dict[str, object]:
    config = TrainConfig(
        model=args.model,
        weights=args.weights,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        freeze_backbone=args.freeze_backbone,
        show_plots=args.show_plots,
    )
    metrics = run_training(config)
    print("Pelatihan Deep Learning selesai.")
    return metrics


def _handle_dl_eval(args: argparse.Namespace) -> dict[str, object]:
    config = EvalConfig(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
    )
    metrics = run_evaluation(config)
    print("Evaluasi Deep Learning selesai.")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """Create the root CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ml-project",
        description=(
            "CLI terpadu untuk Linear Regression, Mean Shift Clustering, dan Deep Learning."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_linear_regression_parser(subparsers)
    _add_mean_shift_parser(subparsers)
    _add_deep_learning_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and execute the selected ML workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, indent=2))
