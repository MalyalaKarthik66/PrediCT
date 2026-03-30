"""Dataset utilities for COCA heart segmentation per project blueprint.

Functions build MONAI-friendly data dictionaries, perform patient-level splits,
and compute dataset statistics needed for the evaluation notebook.
"""
from __future__ import annotations

import json
import logging
import math
import pathlib
from typing import Dict, List, Sequence, Tuple, cast

import nibabel as nib
import numpy as np
import pandas as pd
from monai.data.dataloader import DataLoader
from monai.data.dataset import CacheDataset
from monai.transforms.compose import Compose

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_data_dicts(image_dir: str, label_dir: str) -> List[Dict[str, str]]:
    """Create MONAI data dictionaries for all images that have matching labels."""
    image_dir_path = pathlib.Path(image_dir)
    label_dir_path = pathlib.Path(label_dir)
    data_dicts: List[Dict[str, str]] = []

    for image_path in sorted(image_dir_path.glob("*.nii.gz")):
        scan_id = image_path.stem
        label_path = label_dir_path / f"{scan_id}_heart_mask.nii.gz"
        if not label_path.exists():
            LOGGER.warning("No matching label for %s", image_path.name)
            continue
        data_dicts.append({"image": str(image_path), "label": str(label_path)})

    if not data_dicts:
        LOGGER.warning("No image/label pairs found under %s and %s", image_dir, label_dir)
    return data_dicts


def patient_level_split(
    data_dicts: Sequence[Dict[str, str]],
    train_frac: float,
    val_frac: float,
    seed: int = 42,
    split_path: str | pathlib.Path | None = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Split patient list deterministically and save split assignment."""
    rng = np.random.default_rng(seed)
    total = len(data_dicts)
    if total == 0:
        return [], [], []

    if total < 35:
        LOGGER.warning("Dataset has %d scans (<35); using 80/10/10 split per blueprint", total)
        train_frac, val_frac = 0.8, 0.1

    indices = np.arange(total)
    rng.shuffle(indices)
    n_train = math.floor(total * train_frac)
    n_val = math.floor(total * val_frac)
    n_test = total - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_list = [data_dicts[i] for i in train_idx]
    val_list = [data_dicts[i] for i in val_idx]
    test_list = [data_dicts[i] for i in test_idx]

    if split_path:
        split_records = {}
        for i in train_idx:
            split_records[pathlib.Path(data_dicts[i]["image"]).stem] = "train"
        for i in val_idx:
            split_records[pathlib.Path(data_dicts[i]["image"]).stem] = "val"
        for i in test_idx:
            split_records[pathlib.Path(data_dicts[i]["image"]).stem] = "test"
        split_path = pathlib.Path(split_path)
        _ensure_dir(split_path.parent)
        with split_path.open("w", encoding="utf-8") as f:
            json.dump(split_records, f, indent=2)

    return train_list, val_list, test_list


def get_dataloaders(
    train_data: Sequence[Dict[str, str]],
    val_data: Sequence[Dict[str, str]],
    train_transforms: Compose,
    val_transforms: Compose,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create CacheDataset-backed DataLoaders for training and validation."""
    train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0, num_workers=2)
    val_ds = CacheDataset(data=val_data, transform=val_transforms, cache_rate=1.0, num_workers=2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    return train_loader, val_loader


def compute_dataset_statistics(data_dicts: Sequence[Dict[str, str]]) -> pd.DataFrame:
    """Compute spacing and heart voxel percentage statistics for each scan."""
    records = []
    for entry in data_dicts:
        image_path = pathlib.Path(entry["image"])
        label_path = pathlib.Path(entry["label"])
        img = cast(nib.Nifti1Image, nib.load(str(image_path)))
        spacing = cast(Tuple[float, float, float], tuple(img.header.get_zooms()[:3]))
        spacing_x, spacing_y, spacing_z = spacing
        mask = cast(nib.Nifti1Image, nib.load(str(label_path))).get_fdata()
        voxel_count = mask.size
        heart_voxels = np.count_nonzero(mask)
        heart_pct = heart_voxels / float(voxel_count) if voxel_count else 0.0
        records.append(
            {
                "scan_id": image_path.stem,
                "spacing_x": spacing_x,
                "spacing_y": spacing_y,
                "spacing_z": spacing_z,
                "heart_voxel_pct": heart_pct,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        LOGGER.info("Dataset statistics:\n%s", df.describe())
    else:
        LOGGER.warning("No records available to compute statistics")
    return df
