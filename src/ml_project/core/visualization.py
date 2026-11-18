"""Shared plotting helpers and cross-platform image viewer utilities."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def save_figure(figure: Figure, output_path: Path, dpi: int = 160) -> Path:
    """Apply tight layout, save a Matplotlib figure, and release memory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


def save_correlation_heatmap(
    frame: pd.DataFrame,
    output_path: Path,
    title: str = "Korelasi Fitur Wine Quality",
) -> Path:
    """Generate and save a correlation matrix heatmap for numeric columns."""
    correlation = frame.corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(14, 11))
    sns.heatmap(correlation, cmap="coolwarm", center=0, square=True, ax=axis)
    axis.set_title(title)
    return save_figure(figure, output_path)


def open_saved_plots(output_dir: Path, filenames: Sequence[str]) -> None:
    """Open generated PNG charts using the system's default image viewer."""
    image_paths = [
        output_dir / filename for filename in filenames if (output_dir / filename).exists()
    ]
    if not image_paths:
        return

    if sys.platform.startswith("linux"):
        opener = shutil.which("xdg-open")
        if opener is None:
            print(f"Tidak menemukan xdg-open. Grafik tersedia di: {output_dir}")
            return
        for image_path in image_paths:
            subprocess.Popen(
                [opener, str(image_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    elif sys.platform == "darwin":
        for image_path in image_paths:
            subprocess.Popen(
                ["open", str(image_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    elif sys.platform == "win32":
        for image_path in image_paths:
            os.startfile(image_path)  # type: ignore[attr-defined]
    else:
        print(f"Buka grafik secara manual dari: {output_dir}")
        return

    print(f"Membuka {len(image_paths)} grafik PNG dengan aplikasi bawaan sistem.")
