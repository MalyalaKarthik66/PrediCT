"""DataLoader construction for CAC segmentation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from torch.utils.data import DataLoader

from predict_cac.datasets.coca_dataset import COCADataset


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
