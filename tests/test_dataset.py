from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict_cac.datasets.coca_dataset import create_train_val_test_splits
from predict_cac.datasets.patch_sampler import sample_patch_centers, sample_patch_pair


def test_patch_sampler_returns_points() -> None:
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[5, 5, 5] = 1
    centers = sample_patch_centers(mask, n_samples=4)
    assert len(centers) > 0
    assert all(len(c) == 3 for c in centers)


def test_sample_patch_pair_respects_patch_size() -> None:
    image = np.zeros((16, 16, 16), dtype=np.float32)
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[8, 8, 8] = 1
    image_patch, mask_patch = sample_patch_pair(
        image,
        mask,
        patch_size=(8, 8, 8),
        positive_probability=1.0,
        rng=np.random.default_rng(42),
    )
    assert image_patch.shape == (8, 8, 8)
    assert mask_patch.shape == (8, 8, 8)
    assert int(mask_patch.max()) == 1


def test_create_train_val_test_splits_fallback_without_agatston(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    for idx in range(10):
        (image_dir / f"scan_{idx}.nii.gz").write_text("x", encoding="utf-8")
        (mask_dir / f"scan_{idx}.nii.gz").write_text("x", encoding="utf-8")

    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "scan_id": [f"scan_{idx}" for idx in range(10)],
            "spacing_x": [1.0] * 10,
        }
    ).to_csv(metadata_csv, index=False)

    splits = create_train_val_test_splits(
        image_dir=image_dir,
        mask_dir=mask_dir,
        metadata_csv=metadata_csv,
        seed=42,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == 10
    assert len(splits["train"]) >= 6
