"""Utilities to dilate sparse centerline masks into vessel zones."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def dilate_centerline_mask(mask: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Dilate a binary centerline mask to approximate vessel territory."""
    return binary_dilation(mask > 0, iterations=iterations).astype(np.uint8)
