from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict_cac.transforms.hu_window import clip_hu
from predict_cac.transforms.normalization import zscore_normalize
from predict_cac.transforms.roi_crop import crop_to_nonzero_bbox


def test_clip_hu_preserves_in_range_values() -> None:
    vol = np.array([-150.0, 0.0, 500.0], dtype=np.float32)
    out = clip_hu(vol, -200.0, 1000.0)
    assert np.allclose(out, vol)


def test_zscore_std_close_to_one() -> None:
    vol = np.linspace(0.0, 10.0, num=101, dtype=np.float32)
    out = zscore_normalize(vol)
    assert abs(float(out.std()) - 1.0) < 1e-5


def test_crop_to_nonzero_bbox_reduces_shape() -> None:
    vol = np.zeros((20, 20, 20), dtype=np.float32)
    vol[8:12, 9:13, 10:14] = 1.0
    cropped = crop_to_nonzero_bbox(vol, margin=0)
    assert cropped.shape == (4, 4, 4)


def test_crop_to_nonzero_bbox_no_nonzero_returns_original() -> None:
    vol = np.zeros((5, 6, 7), dtype=np.float32)
    cropped = crop_to_nonzero_bbox(vol, margin=2)
    assert cropped.shape == vol.shape
