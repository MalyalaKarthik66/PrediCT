"""Centerline point transform utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from predict_cac.utils.io import ensure_dir


def transform_centerline_points(
    points_xyz: np.ndarray,
    transform_path: Path,
    out_path: Path,
) -> Path:
    """Apply a saved transform to centerline XYZ points."""
    tx = sitk.ReadTransform(str(transform_path))
    warped = np.array([tx.TransformPoint(tuple(map(float, p))) for p in points_xyz], dtype=np.float32)
    ensure_dir(out_path.parent)
    np.save(out_path, warped)
    return out_path
