"""DataLoader construction for CAC segmentation experiments."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from monai.data import CacheDataset
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged
from torch.utils.data import DataLoader

from predict_cac.datasets.coca_dataset import COCADataset, create_train_val_test_splits
from predict_cac.datasets.patch_sampler import sample_patch_pair
from predict_cac.transforms.augmentations import build_training_augment_fn


def build_dataloader(
    image_dir: Path,
    mask_dir: Path,
    batch_size: int = 1,
    shuffle: bool = True,
    transform: Callable | None = None,
) -> DataLoader:
    """Build a PyTorch DataLoader for volume/mask pairs."""
    image_paths = sorted(image_dir.glob('*.nii.gz'))
    mask_paths = sorted(mask_dir.glob('*.nii.gz'))
    dataset = COCADataset(image_paths=image_paths, mask_paths=mask_paths, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _base_cache_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),
            EnsureTyped(keys=["image", "mask"], dtype=torch.float32),
        ]
    )


def _patch_collate(
    batch: list[dict[str, Any]],
    patch_size: tuple[int, int, int],
    positive_probability: float,
    train_augment_fn: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None,
) -> dict[str, torch.Tensor]:
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    rng = np.random.default_rng()

    for item in batch:
        image = np.asarray(item["image"][0], dtype=np.float32)
        mask = np.asarray(item["mask"][0], dtype=np.float32)
        image_patch, mask_patch = sample_patch_pair(
            image,
            mask,
            patch_size=patch_size,
            positive_probability=positive_probability,
            rng=rng,
        )

        if train_augment_fn is not None:
            augmented = train_augment_fn({"image": image_patch, "mask": mask_patch})
            image_patch = np.asarray(augmented["image"], dtype=np.float32)
            mask_patch = np.asarray(augmented["mask"], dtype=np.float32)

        images.append(torch.from_numpy(image_patch).float().unsqueeze(0))
        masks.append(torch.from_numpy((mask_patch > 0).astype(np.float32)).unsqueeze(0))

    return {
        "image": torch.stack(images, dim=0),
        "mask": torch.stack(masks, dim=0),
    }


def _cache_dataset_from_records(records: list[dict[str, object]], cache_rate: float = 1.0) -> CacheDataset:
    monai_items = [{"image": str(rec["image_path"]), "mask": str(rec["mask_path"])} for rec in records]
    return CacheDataset(data=monai_items, transform=_base_cache_transforms(), cache_rate=float(cache_rate), num_workers=0)


def build_segmentation_dataloaders(
    config: Mapping[str, Any],
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders using MONAI CacheDataset and patch sampling."""
    data_cfg = config.get("data", {})
    split_cfg = config.get("split", {})
    loader_cfg = config.get("loader", {})
    aug_cfg = config.get("augmentations", {})

    image_dir = Path(str(data_cfg.get("image_dir", "data/preprocessed")))
    mask_dir = Path(str(data_cfg.get("mask_dir", "data/masks")))
    metadata_csv_raw = data_cfg.get("metadata_csv", "data/metadata.csv")
    metadata_csv = Path(str(metadata_csv_raw)) if metadata_csv_raw else None

    split_seed = int(split_cfg.get("seed", 42))
    train_ratio = float(split_cfg.get("train", 0.70))
    val_ratio = float(split_cfg.get("val", 0.15))
    test_ratio = float(split_cfg.get("test", 0.15))

    patch_size_list = loader_cfg.get("patch_size", [64, 64, 32])
    patch_size = (int(patch_size_list[0]), int(patch_size_list[1]), int(patch_size_list[2]))
    batch_size = int(loader_cfg.get("batch_size", 1))
    num_workers = int(loader_cfg.get("num_workers", 0))
    cache_rate = float(loader_cfg.get("cache_rate", 1.0))
    positive_patch_probability = float(loader_cfg.get("positive_patch_probability", 0.8))
    val_positive_patch_probability = float(loader_cfg.get("val_positive_patch_probability", 0.5))

    splits = create_train_val_test_splits(
        image_dir=image_dir,
        mask_dir=mask_dir,
        metadata_csv=metadata_csv,
        seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_aug_fn = build_training_augment_fn(aug_cfg if bool(aug_cfg.get("enabled", False)) else None)

    train_ds = _cache_dataset_from_records(splits["train"], cache_rate=cache_rate)
    val_ds = _cache_dataset_from_records(splits["val"], cache_rate=cache_rate)
    test_ds = _cache_dataset_from_records(splits["test"], cache_rate=cache_rate)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: _patch_collate(
            batch,
            patch_size=patch_size,
            positive_probability=positive_patch_probability,
            train_augment_fn=train_aug_fn,
        ),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: _patch_collate(
            batch,
            patch_size=patch_size,
            positive_probability=val_positive_patch_probability,
            train_augment_fn=None,
        ),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: _patch_collate(
            batch,
            patch_size=patch_size,
            positive_probability=val_positive_patch_probability,
            train_augment_fn=None,
        ),
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
