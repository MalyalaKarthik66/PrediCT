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


def sample_foreground_biased_center(
    mask: np.ndarray,
    positive_probability: float = 0.8,
    rng: np.random.Generator | None = None,
) -> tuple[int, int, int]:
    """Sample one center with configurable foreground bias.

    With probability ``positive_probability`` the center is sampled from positive
    mask voxels when available, otherwise from background voxels.
    """
    if mask.ndim != 3:
        raise ValueError("mask must be 3D")

    generator = rng if rng is not None else np.random.default_rng()
    positive_probability = float(np.clip(positive_probability, 0.0, 1.0))

    pos = np.argwhere(mask > 0)
    neg = np.argwhere(mask <= 0)

    choose_positive = bool(generator.random() < positive_probability and len(pos) > 0)
    pool = pos if choose_positive else neg
    if len(pool) == 0:
        pool = pos if len(pos) > 0 else np.argwhere(np.ones_like(mask, dtype=np.uint8) > 0)

    index = int(generator.integers(low=0, high=len(pool)))
    return tuple(int(v) for v in pool[index])


def extract_patch(volume: np.ndarray, center: tuple[int, int, int], patch_size: tuple[int, int, int]) -> np.ndarray:
    """Extract a patch centered at ``center`` with zero-padding at boundaries."""
    if volume.ndim != 3:
        raise ValueError("volume must be 3D")

    patch_size_arr = np.asarray(patch_size, dtype=int)
    if np.any(patch_size_arr <= 0):
        raise ValueError("patch_size values must be > 0")

    half = patch_size_arr // 2
    start = np.asarray(center, dtype=int) - half
    end = start + patch_size_arr

    vol_shape = np.asarray(volume.shape, dtype=int)
    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, vol_shape)

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    patch = np.zeros(tuple(int(v) for v in patch_size_arr), dtype=volume.dtype)
    patch[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = volume[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]
    return patch


def sample_patch_pair(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: tuple[int, int, int],
    positive_probability: float = 0.8,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample aligned image/mask patches with foreground-biased center selection."""
    if image.shape != mask.shape:
        raise ValueError("image and mask must share the same shape")

    center = sample_foreground_biased_center(mask, positive_probability=positive_probability, rng=rng)
    return extract_patch(image, center=center, patch_size=patch_size), extract_patch(mask, center=center, patch_size=patch_size)
