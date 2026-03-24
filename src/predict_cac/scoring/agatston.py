"""Agatston score computation for CAC masks."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _density_factor(max_hu: float) -> int:
    if max_hu < 130:
        return 0
    if max_hu < 200:
        return 1
    if max_hu < 300:
        return 2
    if max_hu < 400:
        return 3
    return 4


def compute_agatston_score(
    ct_hu: np.ndarray,
    calcium_mask: np.ndarray,
    pixel_spacing_mm: tuple[float, float] = (1.0, 1.0),
) -> dict[str, float]:
    """Compute lesion area, density-weighted lesion score, and final Agatston score."""
    binary = calcium_mask > 0
    labeled, n = ndimage.label(binary)
    pixel_area = float(pixel_spacing_mm[0] * pixel_spacing_mm[1])

    total_score = 0.0
    total_area = 0.0

    for label_id in range(1, n + 1):
        lesion = labeled == label_id
        area = float(lesion.sum() * pixel_area)
        max_hu = float(ct_hu[lesion].max()) if lesion.any() else 0.0
        factor = _density_factor(max_hu)
        total_area += area
        total_score += area * factor

    mean_density = float(ct_hu[binary].mean()) if binary.any() else 0.0
    return {
        "lesion_area_mm2": total_area,
        "calcium_density_hu": mean_density,
        "agatston_score": total_score,
    }
