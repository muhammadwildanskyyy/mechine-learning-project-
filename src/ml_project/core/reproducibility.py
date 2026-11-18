"""Reproducibility and compute-device selection utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve the execution device (auto selects CUDA, then MPS, then CPU)."""
    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Device 'cuda' diminta tetapi CUDA tidak tersedia.")
    if normalized == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("Device 'mps' diminta tetapi MPS tidak tersedia.")
    if normalized not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Device tidak dikenali: '{requested}'. Pilihan: auto, cpu, cuda, mps.")
    return torch.device(normalized)
