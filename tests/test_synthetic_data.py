from pathlib import Path
import sys
from typing import cast

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict_cac.evaluation.proximity_metrics import calcium_distance_summary
from predict_cac.utils.synthetic_data import generate_synthetic_demo_data


def test_generate_synthetic_demo_data_outputs_exist() -> None:
    paths = generate_synthetic_demo_data(ROOT, seed=42)
    assert paths.fixed_nifti.exists()
    assert paths.moving_nifti.exists()
    assert paths.calcium_mask.exists()
    assert paths.centerline_mask.exists()


def test_generate_synthetic_demo_data_masks_non_empty() -> None:
    paths = generate_synthetic_demo_data(ROOT, seed=43)
    calcium_img = cast(nib.Nifti1Image, nib.load(str(paths.calcium_mask)))
    centerline_img = cast(nib.Nifti1Image, nib.load(str(paths.centerline_mask)))
    calcium = np.asarray(calcium_img.get_fdata(), dtype=np.float32)
    centerline = np.asarray(centerline_img.get_fdata(), dtype=np.float32)
    assert float(calcium.sum()) > 0.0
    assert float(centerline.sum()) > 0.0


def test_synthetic_proximity_reasonable_range() -> None:
    paths = generate_synthetic_demo_data(ROOT, seed=44)
    calcium_img = cast(nib.Nifti1Image, nib.load(str(paths.calcium_mask)))
    centerline_img = cast(nib.Nifti1Image, nib.load(str(paths.centerline_mask)))
    calcium = np.asarray(calcium_img.get_fdata(), dtype=np.float32)
    centerline = np.asarray(centerline_img.get_fdata(), dtype=np.float32)
    summary = calcium_distance_summary(calcium, centerline, spacing_xyz=(1.0, 1.0, 1.0), threshold_mm=10.0)
    assert 0.0 <= summary["percent_within_10mm"] <= 100.0
    assert summary["mean_distance_mm"] >= 0.0
