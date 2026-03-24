"""Cardiac region-of-interest cropping helpers."""

from __future__ import annotations

import numpy as np


def crop_to_nonzero_bbox(volume: np.ndarray, margin: int = 8) -> np.ndarray:
    """Crop volume to non-zero bounding box with optional margin."""
    coords = np.argwhere(volume != 0)
    if coords.size == 0:
        return volume

    min_zyx = coords.min(axis=0)
    max_zyx = coords.max(axis=0) + 1

    min_zyx = np.maximum(min_zyx - margin, 0)
    max_zyx = np.minimum(max_zyx + margin, np.array(volume.shape))

    return volume[
        min_zyx[0]:max_zyx[0],
        min_zyx[1]:max_zyx[1],
        min_zyx[2]:max_zyx[2],
    ]


def crop_with_mask(volume: np.ndarray, roi_mask: np.ndarray, margin: int = 8) -> np.ndarray:
    """Crop volume to ROI mask bounding box."""
    if roi_mask.shape != volume.shape:
        raise ValueError("roi_mask must have same shape as volume")
    return crop_to_nonzero_bbox((roi_mask > 0).astype(np.uint8), margin=margin) * volume
