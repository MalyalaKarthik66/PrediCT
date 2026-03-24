from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict_cac.datasets.patch_sampler import sample_patch_centers


def test_patch_sampler_returns_points() -> None:
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[5, 5, 5] = 1
    centers = sample_patch_centers(mask, n_samples=4)
    assert len(centers) > 0
    assert all(len(c) == 3 for c in centers)
