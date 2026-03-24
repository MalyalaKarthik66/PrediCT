from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict_cac.transforms.hu_window import clip_hu
from predict_cac.transforms.normalization import zscore_normalize


def test_hu_clip_range() -> None:
    vol = np.array([-500.0, 0.0, 1200.0], dtype=np.float32)
    out = clip_hu(vol, -200.0, 1000.0)
    assert float(out.min()) >= -200.0
    assert float(out.max()) <= 1000.0


def test_zscore_properties() -> None:
    vol = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = zscore_normalize(vol)
    assert abs(float(out.mean())) < 1e-5
