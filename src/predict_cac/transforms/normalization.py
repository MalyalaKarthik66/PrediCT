"""Intensity normalization transforms."""

from __future__ import annotations

import numpy as np


def zscore_normalize(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply z-score normalization to a CT volume."""
    mean = float(volume.mean())
    std = float(volume.std())
    return (volume - mean) / (std + eps)
