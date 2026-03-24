"""Patch sampling utilities for sparse CAC voxel distributions."""

from __future__ import annotations

import numpy as np


def sample_patch_centers(mask: np.ndarray, n_samples: int = 8) -> list[tuple[int, int, int]]:
    """Sample patch centers with positive-prioritized strategy."""
    pos = np.argwhere(mask > 0)
    neg = np.argwhere(mask == 0)

    centers: list[tuple[int, int, int]] = []
    n_pos = min(len(pos), max(1, n_samples // 2))
    n_neg = min(len(neg), n_samples - n_pos)

    if n_pos > 0:
        idx = np.random.choice(len(pos), size=n_pos, replace=len(pos) < n_pos)
        centers.extend([tuple(int(v) for v in pos[i]) for i in idx])

    if n_neg > 0:
        idx = np.random.choice(len(neg), size=n_neg, replace=len(neg) < n_neg)
        centers.extend([tuple(int(v) for v in neg[i]) for i in idx])

    return centers
