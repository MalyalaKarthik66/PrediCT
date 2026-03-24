"""PyTorch dataset for COCA CT and calcium-mask volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, cast

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class COCADataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset that loads CT volume and calcium mask pairs."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        mask_paths: Sequence[Path],
        transform: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None,
    ) -> None:
        if len(image_paths) != len(mask_paths):
            raise ValueError("image_paths and mask_paths must have equal length")
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_nii = cast(nib.Nifti1Image, nib.load(str(self.image_paths[index])))
        mask_nii = cast(nib.Nifti1Image, nib.load(str(self.mask_paths[index])))
        image = np.asarray(image_nii.get_fdata(), dtype=np.float32)
        mask = np.asarray(mask_nii.get_fdata(), dtype=np.float32)

        sample: dict[str, np.ndarray] = {"image": image, "mask": mask}
        if self.transform is not None:
            sample = self.transform(sample)

        return {
            "image": torch.from_numpy(sample["image"]).float().unsqueeze(0),
            "mask": torch.from_numpy(sample["mask"]).float().unsqueeze(0),
        }
