"""MONAI preprocessing and augmentation pipelines per project blueprint.

Train pipeline applies the five mandated augmentations (flip, rotate, noise, zoom,
elastic) after spacing/HU normalization. Validation pipeline is deterministic.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CropForegroundd, RandSpatialCropd
from monai.transforms.intensity.dictionary import RandGaussianNoised, ScaleIntensityRanged
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.post.dictionary import AsDiscreted
from monai.transforms.spatial.dictionary import (
    Orientationd,
    Rand3DElasticd,
    RandFlipd,
    RandRotated,
    RandZoomd,
    Spacingd,
)
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped, ToTensord


def _base_transforms(config: Dict, include_center_crop: bool = True) -> List:
    spacing = config["data"]["spacing"]
    hu_min = config["data"]["hu_min"]
    hu_max = config["data"]["hu_max"]
    roi_size = config["training"]["roi_size"]
    transforms: List = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image"], track_meta=True),
        EnsureTyped(keys=["label"], dtype=torch.long, track_meta=True),
        AsDiscreted(keys=["label"], threshold=0.5),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=hu_min,
            a_max=hu_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # Crop to label foreground to keep heart centered
        CropForegroundd(keys=["image", "label"], source_key="label"),
    ]
    if include_center_crop:
        # Deterministic center crop after foreground detection so heart stays in view
        transforms.append(
            RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_center=False, random_size=False)
        )
    return transforms


def get_train_transforms(config: Dict) -> Compose:
    """Training transforms including all required augmentations."""
    transforms = _base_transforms(config, include_center_crop=True)
    transforms.extend(
        [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandRotated(
                keys=["image", "label"],
                range_x=0.15,
                prob=0.3,
                mode=("bilinear", "nearest"),
            ),
            RandGaussianNoised(keys=["image"], mean=0.0, std=0.01, prob=0.2),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.9,
                max_zoom=1.1,
                prob=0.3,
                mode=("trilinear", "nearest"),
            ),
            Rand3DElasticd(
                keys=["image", "label"],
                sigma_range=(5, 7),
                magnitude_range=(50, 150),
                prob=0.2,
                mode=("bilinear", "nearest"),
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return Compose(transforms)


def get_val_transforms(config: Dict, crop: bool = True) -> Compose:
    """Validation transforms without augmentation (deterministic)."""
    transforms = _base_transforms(config, include_center_crop=crop)
    transforms.append(EnsureTyped(keys=["image", "label"], track_meta=True))
    return Compose(transforms)


def get_inference_transforms(config: Dict) -> Compose:
    """Inference transforms for CT volumes without labels."""
    spacing = config["data"]["spacing"]
    hu_min = config["data"]["hu_min"]
    hu_max = config["data"]["hu_max"]
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=hu_min,
            a_max=hu_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys=["image"], track_meta=True),
    ]
    return Compose(transforms)
