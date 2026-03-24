"""End-to-end preprocessing pipeline for CAC segmentation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, cast

import nibabel as nib
import numpy as np

from predict_cac.transforms.hu_window import clip_hu
from predict_cac.transforms.normalization import zscore_normalize
from predict_cac.transforms.resample_spacing import numpy_to_sitk, resample_to_spacing, sitk_to_numpy
from predict_cac.transforms.roi_crop import crop_to_nonzero_bbox
from predict_cac.utils.io import ensure_dir


def _zooms_xyz(image: nib.Nifti1Image) -> tuple[float, float, float]:
    raw = [float(v) for v in image.header.get_zooms()]
    while len(raw) < 3:
        raw.append(1.0)
    return raw[0], raw[1], raw[2]


def _load_volume(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = cast(nib.Nifti1Image, nib.load(str(path)))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    return data, _zooms_xyz(img)


def _save_volume(path: Path, volume: np.ndarray) -> None:
    ensure_dir(path.parent)
    out = nib.Nifti1Image(volume.astype(np.float32), affine=np.eye(4))
    nib.save(out, str(path))


def run_preprocessing_pipeline(
    input_nifti_dir: Path,
    output_dir: Path,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    augment_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[Path]:
    """Run HU clipping, z-score, isotropic resampling, ROI crop, and optional augmentation."""
    ensure_dir(output_dir)
    outputs: list[Path] = []

    for nifti_path in sorted(input_nifti_dir.glob('*.nii.gz')):
        volume, spacing = _load_volume(nifti_path)
        volume = clip_hu(volume, -200.0, 1000.0)
        volume = zscore_normalize(volume)

        sitk_img = numpy_to_sitk(volume, spacing=spacing)
        sitk_resampled = resample_to_spacing(sitk_img, target_spacing=target_spacing, is_mask=False)
        volume = sitk_to_numpy(sitk_resampled)

        volume = crop_to_nonzero_bbox(volume, margin=8)
        if augment_fn is not None:
            volume = augment_fn(volume)

        out_path = output_dir / nifti_path.name
        _save_volume(out_path, volume)
        outputs.append(out_path)

    return outputs
