"""Shared dataset loading and JSON serialization helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_wine_dataset(
    data_path: Path,
    target_column: str,
    *,
    drop_missing_target: bool = True,
    min_rows: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a semicolon-delimited Wine Quality CSV and validate numeric features."""
    frame = pd.read_csv(data_path, sep=";")
    frame.columns = frame.columns.str.strip()

    if frame.empty:
        raise ValueError("Dataset kosong.")
    if target_column not in frame.columns:
        raise ValueError(
            f"Kolom target '{target_column}' tidak ditemukan. "
            f"Kolom tersedia: {', '.join(frame.columns)}"
        )

    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    if drop_missing_target:
        missing_count = int(frame[target_column].isna().sum())
        if missing_count:
            print(f"Menghapus {missing_count} baris dengan target yang kosong/tidak valid.")
            frame = frame.dropna(subset=[target_column]).copy()

    features = frame.drop(columns=target_column)
    non_numeric = features.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Semua fitur harus numerik. Kolom nonnumerik ditemukan: {', '.join(non_numeric)}"
        )
    if features.shape[1] == 0:
        raise ValueError("Dataset tidak memiliki fitur untuk diproses.")
    if len(frame) < min_rows:
        raise ValueError(f"Dataset membutuhkan minimal {min_rows} baris valid.")

    return features, frame[target_column].rename(target_column)


def save_json(payload: Mapping[str, Any], output_path: Path) -> Path:
    """Serialize a mapping as UTF-8 formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return output_path
