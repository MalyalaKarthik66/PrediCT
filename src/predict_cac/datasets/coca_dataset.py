"""PyTorch dataset for COCA CT and calcium-mask volumes."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, cast

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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


def _strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _agatston_column(df: pd.DataFrame) -> str | None:
    for col in ("agatston_score", "agatston", "score", "cac_score"):
        if col in df.columns:
            return col
    return None


def agatston_risk_category(score: float) -> str:
    if score <= 0:
        return "0"
    if score < 100:
        return "1-99"
    if score < 400:
        return "100-399"
    return "400+"


def discover_image_mask_pairs(image_dir: Path, mask_dir: Path) -> list[dict[str, object]]:
    """Match preprocessed images and masks by filename stem."""
    image_paths = sorted(image_dir.glob("*.nii.gz"))
    all_masks = sorted(mask_dir.glob("*.nii.gz"))
    mask_by_stem = {_strip_nii_suffix(path): path for path in all_masks}
    default_mask = all_masks[0] if len(all_masks) == 1 else None

    samples: list[dict[str, object]] = []
    for image_path in image_paths:
        stem = _strip_nii_suffix(image_path)
        mask_path = mask_by_stem.get(stem)
        if mask_path is None and default_mask is not None:
            mask_path = default_mask
        if mask_path is None:
            continue
        samples.append(
            {
                "scan_id": stem,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )
    return samples


def _resolve_metadata_scan_id(frame: pd.DataFrame) -> pd.Series:
    if "scan_id" in frame.columns:
        return frame["scan_id"].astype(str)
    if "nifti_path" in frame.columns:
        return frame["nifti_path"].astype(str).map(lambda raw: _strip_nii_suffix(Path(raw)))
    return pd.Series([""] * len(frame), index=frame.index, dtype="object")


def _stratified_or_random_split(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(frame) < 3:
        return frame.copy(), frame.iloc[0:0].copy(), frame.iloc[0:0].copy()

    stratify_labels = None
    if "strata_label" in frame.columns:
        counts = frame["strata_label"].value_counts()
        if not counts.empty and int(counts.min()) >= 2:
            stratify_labels = frame["strata_label"]

    test_size = float(val_ratio + test_ratio)
    train_df, temp_df = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_labels,
        shuffle=True,
    )

    if temp_df.empty:
        return train_df, temp_df, temp_df

    rel_test = float(test_ratio / (val_ratio + test_ratio)) if (val_ratio + test_ratio) > 0 else 0.5
    temp_stratify = None
    if "strata_label" in temp_df.columns:
        temp_counts = temp_df["strata_label"].value_counts()
        if not temp_counts.empty and int(temp_counts.min()) >= 2:
            temp_stratify = temp_df["strata_label"]

    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        random_state=seed,
        stratify=temp_stratify,
        shuffle=True,
    )
    return train_df, val_df, test_df


def create_train_val_test_splits(
    image_dir: Path,
    mask_dir: Path,
    metadata_csv: Path | None = None,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, list[dict[str, object]]]:
    """Create 70/15/15 splits with Agatston-aware stratification when available.

    If Agatston labels are missing or insufficient for stratification, this
    function falls back to random splitting while preserving deterministic seed.
    """
    samples = discover_image_mask_pairs(image_dir=image_dir, mask_dir=mask_dir)
    if not samples:
        return {"train": [], "val": [], "test": []}

    sample_df = pd.DataFrame(samples)
    sample_df["scan_id"] = sample_df["scan_id"].astype(str)
    sample_df["risk_category"] = "unknown"
    sample_df["agatston_score"] = np.nan

    if metadata_csv is not None and metadata_csv.exists():
        metadata = pd.read_csv(metadata_csv)
        score_col = _agatston_column(metadata)
        if score_col is not None:
            metadata = metadata.copy()
            metadata["scan_id"] = _resolve_metadata_scan_id(metadata)
            metadata[score_col] = pd.to_numeric(metadata[score_col], errors="coerce")
            metadata = metadata.dropna(subset=[score_col])
            if not metadata.empty:
                metadata["risk_category"] = metadata[score_col].map(agatston_risk_category)
                metadata["agatston_quintile"] = pd.qcut(metadata[score_col], q=5, labels=False, duplicates="drop")
                metadata["strata_label"] = (
                    metadata["risk_category"].astype(str)
                    + "_q"
                    + metadata["agatston_quintile"].fillna(-1).astype(int).astype(str)
                )
                merge_cols = ["scan_id", "risk_category", "strata_label", score_col]
                sample_df = sample_df.merge(metadata[merge_cols], on="scan_id", how="left", suffixes=("", "_meta"))
                sample_df["risk_category"] = sample_df["risk_category_meta"].fillna(sample_df["risk_category"])
                sample_df["agatston_score"] = sample_df[score_col].fillna(sample_df["agatston_score"])
                if "strata_label" not in sample_df.columns:
                    sample_df["strata_label"] = np.nan
                sample_df["strata_label"] = sample_df["strata_label"].fillna(sample_df["risk_category"].astype(str))
                sample_df = sample_df.drop(columns=[c for c in ["risk_category_meta", score_col] if c in sample_df.columns])

    train_df, val_df, test_df = _stratified_or_random_split(
        sample_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    keep_cols = ["scan_id", "image_path", "mask_path", "risk_category", "agatston_score"]
    return {
        "train": train_df[keep_cols].to_dict(orient="records"),
        "val": val_df[keep_cols].to_dict(orient="records"),
        "test": test_df[keep_cols].to_dict(orient="records"),
    }
