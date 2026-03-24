"""Utilities for atlas registration and transform serialization."""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from predict_cac.utils.io import ensure_dir


def save_transform(transform: sitk.Transform, out_path: Path) -> Path:
    """Save a SimpleITK transform to disk."""
    ensure_dir(out_path.parent)
    sitk.WriteTransform(transform, str(out_path))
    return out_path


def read_image(path: Path) -> sitk.Image:
    """Read image using SimpleITK."""
    return sitk.ReadImage(str(path))
