"""HU windowing utilities for non-contrast cardiac CT."""

from __future__ import annotations

import numpy as np


def clip_hu(volume: np.ndarray, hu_min: float = -200.0, hu_max: float = 1000.0) -> np.ndarray:
    """Clip Hounsfield units into a clinically relevant CAC window."""
    return np.clip(volume, hu_min, hu_max)
