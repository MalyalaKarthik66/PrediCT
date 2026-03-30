"""Segmentation metrics: Dice, IoU, HD95 (mm), RVE."""
from __future__ import annotations

import json
import pathlib
from typing import Iterable, List, Sequence, Tuple, Union, cast

Spacing3 = Tuple[float, float, float]

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt


def post_pred_sigmoid(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binarize sigmoid probabilities with a fixed threshold (no argmax)."""
    return (prob_map > threshold).astype(np.uint8)


def _dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    return float((2.0 * intersection) / (denom + 1e-8))


def _iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(intersection / (union + 1e-8))


def _normalize_spacing(spacing: Union[Sequence[float], Spacing3, None]) -> Spacing3:
    if spacing is None:
        return (1.0, 1.0, 1.0)
    vals = list(spacing)
    if len(vals) >= 3:
        return (float(vals[0]), float(vals[1]), float(vals[2]))
    if len(vals) == 2:
        return (float(vals[0]), float(vals[1]), 1.0)
    if len(vals) == 1:
        return (float(vals[0]), 1.0, 1.0)
    return (1.0, 1.0, 1.0)


def _hd95(pred: np.ndarray, target: np.ndarray, spacing: Union[Sequence[float], Spacing3]) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    spacing = _normalize_spacing(spacing)
    if not pred.any() or not target.any():
        return float("inf")

    pred_eroded = np.asarray(binary_erosion(pred))
    target_eroded = np.asarray(binary_erosion(target))
    pred_border: np.ndarray = np.logical_xor(pred, pred_eroded)
    target_border: np.ndarray = np.logical_xor(target, target_eroded)
    dt_target = cast(np.ndarray, distance_transform_edt(~target, sampling=spacing))
    dt_pred = cast(np.ndarray, distance_transform_edt(~pred, sampling=spacing))
    distances = np.concatenate([dt_target[pred_border], dt_pred[target_border]])
    if distances.size == 0:
        return float("inf")
    return float(np.percentile(distances, 95))


def _rve(pred: np.ndarray, target: np.ndarray, spacing: Union[Sequence[float], Spacing3]) -> float:
    spacing = _normalize_spacing(spacing)
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    vol_pred = pred.sum() * voxel_volume
    vol_gt = target.sum() * voxel_volume
    return float(abs(vol_pred - vol_gt) / (vol_gt + 1e-8))


def compute_per_case_metrics(
    predictions: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    scan_ids: Sequence[str],
    spacings: Sequence[Union[Sequence[float], Spacing3]],
) -> pd.DataFrame:
    records: List[dict] = []
    for pred, label, scan_id, spacing in zip(predictions, labels, scan_ids, spacings):
        spacing = _normalize_spacing(spacing)
        records.append(
            {
                "scan_id": scan_id,
                "dice": _dice(pred, label),
                "iou": _iou(pred, label),
                "hd95_mm": _hd95(pred, label, spacing),
                "rve": _rve(pred, label, spacing),
            }
        )
    return pd.DataFrame(records)


def compute_summary_statistics(df: pd.DataFrame) -> dict:
    summary = {}
    for metric in ["dice", "iou", "hd95_mm", "rve"]:
        if metric not in df:
            continue
        values = df[metric].to_numpy(dtype=float)
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def save_metrics(df: pd.DataFrame, summary: dict, output_dir: str | pathlib.Path) -> None:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "per_case_metrics.csv", index=False)
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
