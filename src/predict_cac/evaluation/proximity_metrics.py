"""Proximity-based validation metrics for calcium vs coronary centerlines."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _mask_to_points(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask > 0).astype(np.float32)


def calcium_centerline_distances_mm(
    calcium_mask: np.ndarray,
    centerline_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Return nearest-neighbor distance (mm) from each calcium voxel to centerlines."""
    calcium_points = _mask_to_points(calcium_mask)
    centerline_points = _mask_to_points(centerline_mask)

    if len(calcium_points) == 0 or len(centerline_points) == 0:
        return np.array([], dtype=np.float32)

    spacing = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=np.float32)
    calcium_mm = calcium_points * spacing
    centerline_mm = centerline_points * spacing

    tree = cKDTree(centerline_mm)
    distances, _ = tree.query(calcium_mm, workers=-1)
    return distances.astype(np.float32)


def calcium_distance_summary(
    calcium_mask: np.ndarray,
    centerline_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    threshold_mm: float = 10.0,
) -> dict[str, float]:
    """Compute mean, median, and thresholded proximity percentage in mm."""
    distances = calcium_centerline_distances_mm(
        calcium_mask=calcium_mask,
        centerline_mask=centerline_mask,
        spacing_xyz=spacing_xyz,
    )
    if len(distances) == 0:
        return {
            "mean_distance_mm": 0.0,
            "median_distance_mm": 0.0,
            "percent_within_10mm": 0.0,
        }

    return {
        "mean_distance_mm": float(np.mean(distances)),
        "median_distance_mm": float(np.median(distances)),
        "percent_within_10mm": float((distances <= threshold_mm).sum() / len(distances) * 100.0),
    }


def calcium_within_threshold_percent(
    calcium_mask: np.ndarray,
    centerline_mask: np.ndarray,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    threshold_mm: float = 10.0,
) -> float:
    """Compute percentage of calcium voxels within threshold distance of centerlines."""
    summary = calcium_distance_summary(
        calcium_mask=calcium_mask,
        centerline_mask=centerline_mask,
        spacing_xyz=spacing_xyz,
        threshold_mm=threshold_mm,
    )
    return summary["percent_within_10mm"]
