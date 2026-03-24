"""Resampling helpers for isotropic voxel spacing."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import SimpleITK as sitk


def resample_to_spacing(
    image: sitk.Image,
    target_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    is_mask: bool = False,
) -> sitk.Image:
    """Resample SimpleITK image to target spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / float(target_spacing[i]))))
        for i in range(3)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(tuple(float(v) for v in target_spacing))
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return resample.Execute(image)


def numpy_to_sitk(volume: np.ndarray, spacing: Sequence[float]) -> sitk.Image:
    """Convert ndarray (x,y,z) to sitk image and set spacing."""
    if volume.ndim != 3:
        raise ValueError("Expected 3D volume with shape (x, y, z).")
    volume_zyx = np.transpose(volume, (2, 1, 0))
    img = sitk.GetImageFromArray(volume_zyx)
    img.SetSpacing((float(spacing[0]), float(spacing[1]), float(spacing[2])))
    return img


def sitk_to_numpy(image: sitk.Image) -> np.ndarray:
    """Convert sitk image to ndarray (x,y,z)."""
    array_zyx = sitk.GetArrayFromImage(image)
    return np.transpose(array_zyx, (2, 1, 0))
