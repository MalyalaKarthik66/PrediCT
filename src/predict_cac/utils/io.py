"""I/O helpers for the PrediCT CAC pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files_with_suffix(root: Path, suffixes: Iterable[str]) -> list[Path]:
    """Return files under root with one of the provided suffixes."""
    wanted = {s.lower() for s in suffixes}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in wanted]


def stem_without_nii_gz(path: Path) -> str:
    """Return stem for .nii.gz and normal files."""
    if path.name.endswith('.nii.gz'):
        return path.name[:-7]
    return path.stem
