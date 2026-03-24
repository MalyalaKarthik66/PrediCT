"""Synthetic CT and mask generation for CAC pipeline demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from predict_cac.utils.io import ensure_dir


@dataclass
class DemoDataPaths:
    """Output paths for synthetic demo data artifacts."""

    fixed_nifti: Path
    moving_nifti: Path
    fixed_preprocessed: Path
    moving_preprocessed: Path
    calcium_mask: Path
    centerline_mask: Path
    moving_centerline_mask: Path


def _save_nifti(path: Path, data: np.ndarray, spacing_xyz: tuple[float, float, float]) -> Path:
    ensure_dir(path.parent)
    affine = np.diag([spacing_xyz[0], spacing_xyz[1], spacing_xyz[2], 1.0]).astype(np.float32)
    image = nib.Nifti1Image(data.astype(np.float32), affine=affine)
    nib.save(image, str(path))
    return path


def _add_spherical_lesion(
    volume: np.ndarray,
    mask: np.ndarray,
    center_xyz: tuple[int, int, int],
    radius: int,
    hu_value: float,
) -> None:
    x = np.arange(volume.shape[0])[:, None, None]
    y = np.arange(volume.shape[1])[None, :, None]
    z = np.arange(volume.shape[2])[None, None, :]
    cx, cy, cz = center_xyz
    sphere = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius**2
    volume[sphere] = hu_value
    mask[sphere] = 1.0


def _make_centerline_mask(shape_xyz: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape_xyz, dtype=np.float32)
    center_y = shape_xyz[1] // 2
    center_z = shape_xyz[2] // 2
    for x in range(20, shape_xyz[0] - 20):
        y = int(center_y + 10 * np.sin((x - 20) / 15.0))
        z = int(center_z + 6 * np.cos((x - 20) / 20.0))
        y = min(max(y, 0), shape_xyz[1] - 1)
        z = min(max(z, 0), shape_xyz[2] - 1)
        mask[x, y, z] = 1.0
    return mask


def generate_synthetic_demo_data(
    root_dir: Path,
    shape_xyz: tuple[int, int, int] = (128, 128, 64),
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int = 42,
) -> DemoDataPaths:
    """Generate synthetic CT scans, calcium mask, and centerline mask for demo runs."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Baseline soft tissue in HU range around 40 +/- 20.
    fixed = rng.normal(loc=40.0, scale=20.0, size=shape_xyz).astype(np.float32)
    moving = fixed.copy()

    calcium_mask = np.zeros(shape_xyz, dtype=np.float32)
    centerline_mask = _make_centerline_mask(shape_xyz)
    moving_centerline_mask = np.roll(centerline_mask, shift=2, axis=0)

    centerline_points = np.argwhere(centerline_mask > 0)
    lesion_count = 14
    for i in range(lesion_count):
        base = centerline_points[int(rng.integers(0, len(centerline_points)))]

        # Keep most lesions close to centerline and a small subset farther away
        # to emulate realistic imperfect proximity.
        if i < int(lesion_count * 0.8):
            dx = int(rng.integers(-2, 3))
            dy = int(rng.normal(0, 2.0))
            dz = int(rng.normal(0, 2.0))
        else:
            dx = int(rng.integers(-4, 5))
            dy = int(rng.choice([-1, 1]) * rng.integers(10, 16))
            dz = int(rng.choice([-1, 1]) * rng.integers(8, 14))

        cx = int(np.clip(base[0] + dx, 4, shape_xyz[0] - 5))
        cy = int(np.clip(base[1] + dy, 4, shape_xyz[1] - 5))
        cz = int(np.clip(base[2] + dz, 4, shape_xyz[2] - 5))

        hu = float(rng.uniform(850.0, 1200.0))
        radius = int(rng.integers(2, 4))
        _add_spherical_lesion(fixed, calcium_mask, (cx, cy, cz), radius=radius, hu_value=hu)

    # Slight spatially varying intensity perturbation in moving image.
    grad = np.linspace(-1.0, 1.0, num=shape_xyz[0], dtype=np.float32)[:, None, None]
    moving = moving + grad
    moving = np.roll(moving, shift=2, axis=0)

    data_nifti = root_dir / "data" / "nifti"
    data_pre = root_dir / "data" / "preprocessed"
    data_masks = root_dir / "data" / "masks"
    out_center = root_dir / "outputs" / "centerlines_warped"

    paths = DemoDataPaths(
        fixed_nifti=data_nifti / "sample_fixed.nii.gz",
        moving_nifti=data_nifti / "sample_moving.nii.gz",
        fixed_preprocessed=data_pre / "sample_fixed.nii.gz",
        moving_preprocessed=data_pre / "sample_moving.nii.gz",
        calcium_mask=data_masks / "sample_calcium_mask.nii.gz",
        centerline_mask=out_center / "sample_centerline_mask.nii.gz",
        moving_centerline_mask=out_center / "sample_centerline_mask_moving.nii.gz",
    )

    _save_nifti(paths.fixed_nifti, fixed, spacing_xyz)
    _save_nifti(paths.moving_nifti, moving, spacing_xyz)

    # Also store copies in preprocessed so registration can run directly.
    _save_nifti(paths.fixed_preprocessed, fixed, spacing_xyz)
    _save_nifti(paths.moving_preprocessed, moving, spacing_xyz)

    _save_nifti(paths.calcium_mask, calcium_mask, spacing_xyz)
    _save_nifti(paths.centerline_mask, centerline_mask, spacing_xyz)
    _save_nifti(paths.moving_centerline_mask, moving_centerline_mask, spacing_xyz)

    return paths
