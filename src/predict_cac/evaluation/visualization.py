"""Visualization helpers for registration and metric sanity checks."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from predict_cac.evaluation.proximity_metrics import calcium_centerline_distances_mm
from predict_cac.utils.io import ensure_dir


def _load_nifti_array(path: Path) -> np.ndarray:
    image = cast(nib.Nifti1Image, nib.load(str(path)))
    return np.asarray(image.get_fdata())


def save_calcium_overlay_plot(
    image_path: Path,
    calcium_mask_path: Path,
    out_png: Path,
    slice_index: int | None = None,
) -> Path:
    """Save CT slice with calcium-mask overlay."""
    image = _load_nifti_array(image_path)
    calcium = _load_nifti_array(calcium_mask_path)

    z = slice_index if slice_index is not None else image.shape[2] // 2

    ensure_dir(out_png.parent)
    plt.figure(figsize=(7, 7))
    plt.imshow(image[:, :, z], cmap="gray")
    plt.imshow(np.ma.masked_where(calcium[:, :, z] <= 0, calcium[:, :, z]), cmap="autumn", alpha=0.5)
    plt.title("CT + Calcium Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


def save_centerline_overlay_plot(
    image_path: Path,
    centerline_mask_path: Path,
    out_png: Path,
    slice_index: int | None = None,
) -> Path:
    """Save CT slice with transformed-centerline overlay."""
    image = _load_nifti_array(image_path)
    centerline = _load_nifti_array(centerline_mask_path)

    z = slice_index if slice_index is not None else image.shape[2] // 2

    ensure_dir(out_png.parent)
    plt.figure(figsize=(7, 7))
    plt.imshow(image[:, :, z], cmap="gray")
    plt.imshow(np.ma.masked_where(centerline[:, :, z] <= 0, centerline[:, :, z]), cmap="cool", alpha=0.5)
    plt.title("CT + Transformed Centerlines")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


def save_distance_histogram_plot(
    calcium_mask_path: Path,
    centerline_mask_path: Path,
    out_png: Path,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    """Save histogram of calcium-to-centerline distances in millimeters."""
    calcium = _load_nifti_array(calcium_mask_path)
    centerline = _load_nifti_array(centerline_mask_path)
    distances = calcium_centerline_distances_mm(calcium, centerline, spacing_xyz=spacing_xyz)

    ensure_dir(out_png.parent)
    plt.figure(figsize=(7, 5))
    if len(distances) > 0:
        plt.hist(distances, bins=30, color="#2f6690", alpha=0.9)
    plt.xlabel("Distance to centerline (mm)")
    plt.ylabel("Calcium voxel count")
    plt.title("Calcium Distance Distribution")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


def save_registration_plots(
    image_path: Path,
    calcium_mask_path: Path,
    centerline_mask_path: Path,
    out_dir: Path,
) -> dict[str, str]:
    """Create all required registration-evaluation plots and return their paths."""
    ensure_dir(out_dir)
    calcium_overlay = save_calcium_overlay_plot(
        image_path=image_path,
        calcium_mask_path=calcium_mask_path,
        out_png=out_dir / "ct_calcium_overlay.png",
    )
    centerline_overlay = save_centerline_overlay_plot(
        image_path=image_path,
        centerline_mask_path=centerline_mask_path,
        out_png=out_dir / "centerline_overlay.png",
    )

    image_nifti = cast(nib.Nifti1Image, nib.load(str(image_path)))
    zooms = image_nifti.header.get_zooms()[:3]
    histogram = save_distance_histogram_plot(
        calcium_mask_path=calcium_mask_path,
        centerline_mask_path=centerline_mask_path,
        out_png=out_dir / "distance_histogram.png",
        spacing_xyz=(float(zooms[0]), float(zooms[1]), float(zooms[2])),
    )
    return {
        "ct_calcium_overlay": str(calcium_overlay),
        "centerline_overlay": str(centerline_overlay),
        "distance_histogram": str(histogram),
    }
