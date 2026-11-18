"""Filesystem path resolution helpers for datasets and experiment outputs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_WINE_DATASETS: tuple[Path, ...] = (
    DEFAULT_DATA_DIR / "wine-quality" / "winequality-white.csv",
    DEFAULT_DATA_DIR / "wine-quality" / "winequality-red.csv",
    PROJECT_ROOT / "wine+quality" / "winequality-white.csv",
    PROJECT_ROOT / "wine+quality" / "winequality-red.csv",
)

DEFAULT_IMAGE_DATASETS: tuple[Path, ...] = (
    DEFAULT_DATA_DIR / "imagenette2",
    DEFAULT_DATA_DIR / "imagenette2-320",
    DEFAULT_DATA_DIR / "imagenette2-160",
)


def resolve_wine_dataset_path(
    requested_path: Path | None,
    candidates: tuple[Path, ...] = DEFAULT_WINE_DATASETS,
) -> Path:
    """Resolve the requested Wine Quality CSV or fall back to the first available default."""
    if requested_path is not None:
        if requested_path.is_file():
            return requested_path.resolve()
        raise FileNotFoundError(f"Dataset tidak ditemukan: {requested_path}")

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "Dataset Wine Quality tidak ditemukan. Letakkan file CSV pada salah satu "
        f"lokasi berikut atau berikan --data-path:\n{searched}"
    )


def _has_image_splits(dataset_dir: Path) -> bool:
    """Return True when dataset_dir contains non-empty train/ and val/ subdirectories."""
    return (dataset_dir / "train").is_dir() and (dataset_dir / "val").is_dir()


def resolve_image_dataset_dir(
    requested_dir: Path | None,
    candidates: tuple[Path, ...] = DEFAULT_IMAGE_DATASETS,
) -> Path:
    """Resolve an ImageFolder dataset directory containing train/ and val/ splits."""
    if requested_dir is not None:
        if not requested_dir.is_dir():
            raise FileNotFoundError(f"Direktori dataset tidak ditemukan: {requested_dir}")
        if not _has_image_splits(requested_dir):
            raise ValueError(
                f"Direktori dataset '{requested_dir}' harus memiliki subfolder 'train/' dan 'val/'."
            )
        return requested_dir.resolve()

    for candidate in candidates:
        if candidate.is_dir() and _has_image_splits(candidate):
            return candidate.resolve()

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "Dataset gambar (dengan subfolder train/ dan val/) tidak ditemukan. "
        f"Letakkan dataset di salah satu lokasi berikut atau berikan --data-dir:\n{searched}"
    )


def create_run_dir(base_dir: Path, run_name: str | None = None) -> Path:
    """Create and return a unique timestamped experiment directory."""
    base_dir.mkdir(parents=True, exist_ok=True)
    if run_name:
        run_dir = base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir.resolve()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_dir / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{timestamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()
