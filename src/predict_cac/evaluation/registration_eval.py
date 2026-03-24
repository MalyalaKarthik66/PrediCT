"""Registration validation runners and metric persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree

from predict_cac.utils.io import ensure_dir


def _mask_indices_to_physical_points(image: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(image)
    idx_zyx = np.argwhere(arr > 0)
    if len(idx_zyx) == 0:
        return np.empty((0, 3), dtype=np.float64)
    points = [image.TransformIndexToPhysicalPoint((int(x), int(y), int(z))) for z, y, x in idx_zyx]
    return np.asarray(points, dtype=np.float64)


def evaluate_registration(
    calcium_mask_path: Path,
    centerline_mask_path: Path,
    threshold_mm: float = 10.0,
) -> dict[str, Any]:
    """Evaluate centerline proximity metric for one scan."""
    cal_img = sitk.ReadImage(str(calcium_mask_path))
    ctr_img = sitk.ReadImage(str(centerline_mask_path))

    calcium_points = _mask_indices_to_physical_points(cal_img)
    centerline_points = _mask_indices_to_physical_points(ctr_img)

    if len(calcium_points) == 0 or len(centerline_points) == 0:
        summary = {
            "mean_distance_mm": 0.0,
            "median_distance_mm": 0.0,
            "percent_within_10mm": 0.0,
        }
    else:
        tree = cKDTree(centerline_points)
        distances, _ = tree.query(calcium_points, workers=-1)
        summary = {
            "mean_distance_mm": float(np.mean(distances)),
            "median_distance_mm": float(np.median(distances)),
            "percent_within_10mm": float((distances <= threshold_mm).sum() / len(distances) * 100.0),
        }

    return {
        "scan_id": calcium_mask_path.stem,
        "threshold_mm": threshold_mm,
        "mean_distance_mm": summary["mean_distance_mm"],
        "median_distance_mm": summary["median_distance_mm"],
        "percent_within_10mm": summary["percent_within_10mm"],
    }


def save_metrics_json(result: dict[str, Any], metrics_path: Path) -> Path:
    """Save registration metrics as JSON."""
    ensure_dir(metrics_path.parent)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return metrics_path
