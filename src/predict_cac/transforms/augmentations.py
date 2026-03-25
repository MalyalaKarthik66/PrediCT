"""Training-time MONAI augmentation builders for CAC segmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    SqueezeDimd,
)


def _as_bool(mapping: Mapping[str, Any], key: str, default: bool) -> bool:
    return bool(mapping.get(key, default))


def _as_float(mapping: Mapping[str, Any], key: str, default: float) -> float:
    return float(mapping.get(key, default))


def build_training_augment_fn(
    augmentation_cfg: Mapping[str, Any] | None,
) -> Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None:
    """Build MONAI dictionary augmentations for paired image/mask arrays.

    Input and output arrays are expected in shape ``(x, y, z)`` for both
    ``image`` and ``mask`` keys. Internally, MONAI operates channel-first and
    then squeezes back to 3D arrays.
    """
    if augmentation_cfg is None or not _as_bool(augmentation_cfg, "enabled", False):
        return None

    rotation_deg = _as_float(augmentation_cfg, "rotation_deg", 15.0)
    rotation_rad = float(np.deg2rad(rotation_deg))
    scale_range = _as_float(augmentation_cfg, "scale_range", 0.10)

    transforms: list[Any] = [EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel")]
    transforms.append(
        RandAffined(
            keys=["image", "mask"],
            prob=_as_float(augmentation_cfg, "random_affine_prob", 0.7),
            rotate_range=(rotation_rad, rotation_rad, rotation_rad),
            scale_range=(scale_range, scale_range, scale_range),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        )
    )

    elastic_cfg = augmentation_cfg.get("elastic", {})
    if isinstance(elastic_cfg, Mapping) and _as_bool(elastic_cfg, "enabled", True):
        sigma_values = elastic_cfg.get("sigma_range", [4.0, 6.0])
        magnitude_values = elastic_cfg.get("magnitude_range", [50.0, 100.0])
        sigma_range = (float(sigma_values[0]), float(sigma_values[1]))
        magnitude_range = (float(magnitude_values[0]), float(magnitude_values[1]))
        transforms.append(
            Rand3DElasticd(
                keys=["image", "mask"],
                prob=_as_float(elastic_cfg, "prob", 0.2),
                sigma_range=sigma_range,
                magnitude_range=magnitude_range,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            )
        )

    noise_cfg = augmentation_cfg.get("gaussian_noise", {})
    if isinstance(noise_cfg, Mapping) and _as_bool(noise_cfg, "enabled", True):
        transforms.append(
            RandGaussianNoised(
                keys=["image"],
                prob=_as_float(noise_cfg, "prob", 0.2),
                std=_as_float(noise_cfg, "std", 0.01),
            )
        )

    gamma_cfg = augmentation_cfg.get("random_gamma", {})
    if isinstance(gamma_cfg, Mapping) and _as_bool(gamma_cfg, "enabled", True):
        gamma_values = gamma_cfg.get("gamma_range", [0.7, 1.5])
        gamma_range = (float(gamma_values[0]), float(gamma_values[1]))
        transforms.append(
            RandAdjustContrastd(
                keys=["image"],
                prob=_as_float(gamma_cfg, "prob", 0.2),
                gamma=gamma_range,
            )
        )

    transforms.extend([SqueezeDimd(keys=["image", "mask"], dim=0)])
    monai_compose = Compose(transforms)

    def _augment(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        augmented = monai_compose(sample)
        return {
            "image": np.asarray(augmented["image"], dtype=np.float32),
            "mask": np.asarray(augmented["mask"], dtype=np.float32),
        }

    return _augment
