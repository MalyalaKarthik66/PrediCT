from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict_cac.registration.dilate_centerlines import dilate_centerline_mask


def test_centerline_dilation_increases_voxels() -> None:
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[8, 8, 8] = 1
    dilated = dilate_centerline_mask(mask, iterations=2)
    assert int(dilated.sum()) > int(mask.sum())
