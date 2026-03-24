from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict_cac.evaluation.proximity_metrics import (
    calcium_centerline_distances_mm,
    calcium_distance_summary,
)


def test_distance_summary_empty_calcium_returns_zeros() -> None:
    calcium = np.zeros((8, 8, 8), dtype=np.uint8)
    centerline = np.zeros((8, 8, 8), dtype=np.uint8)
    centerline[4, 4, 4] = 1
    summary = calcium_distance_summary(calcium, centerline, threshold_mm=10.0)
    assert summary["mean_distance_mm"] == 0.0
    assert summary["median_distance_mm"] == 0.0
    assert summary["percent_within_10mm"] == 0.0


def test_distance_summary_identical_masks_is_zero_distance() -> None:
    calcium = np.zeros((8, 8, 8), dtype=np.uint8)
    centerline = np.zeros((8, 8, 8), dtype=np.uint8)
    calcium[3, 3, 3] = 1
    centerline[3, 3, 3] = 1
    summary = calcium_distance_summary(calcium, centerline, threshold_mm=10.0)
    assert summary["mean_distance_mm"] == 0.0
    assert summary["median_distance_mm"] == 0.0
    assert summary["percent_within_10mm"] == 100.0


def test_calcium_centerline_distances_mm_single_voxel() -> None:
    calcium = np.zeros((10, 10, 10), dtype=np.uint8)
    centerline = np.zeros((10, 10, 10), dtype=np.uint8)
    calcium[5, 5, 5] = 1
    centerline[5, 5, 7] = 1
    distances = calcium_centerline_distances_mm(calcium, centerline, spacing_xyz=(1.0, 1.0, 1.0))
    assert distances.shape == (1,)
    assert abs(float(distances[0]) - 2.0) < 1e-6


def test_distance_summary_threshold_behavior() -> None:
    calcium = np.zeros((10, 10, 10), dtype=np.uint8)
    centerline = np.zeros((10, 10, 10), dtype=np.uint8)
    calcium[5, 5, 5] = 1
    centerline[5, 5, 9] = 1
    summary = calcium_distance_summary(calcium, centerline, threshold_mm=3.0)
    assert summary["percent_within_10mm"] == 0.0
