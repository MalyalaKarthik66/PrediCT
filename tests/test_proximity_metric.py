from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict_cac.evaluation.proximity_metrics import calcium_within_threshold_percent


def test_proximity_metric_simple_case() -> None:
    calcium = np.zeros((10, 10, 10), dtype=np.uint8)
    centerline = np.zeros((10, 10, 10), dtype=np.uint8)
    calcium[5, 5, 5] = 1
    centerline[5, 5, 6] = 1
    pct = calcium_within_threshold_percent(calcium, centerline, threshold_mm=10.0)
    assert pct == 100.0
