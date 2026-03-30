"""Benchmark model vs TotalSegmentator using pre-generated TS heart masks.

Protocol: same input CTs resampled to 2.5mm, batch=1, two warmup runs, timing
with time.perf_counter plus torch.cuda.synchronize. Measure end-to-end model
runtime (preprocess + inference + postprocess) and TotalSegmentator runtime per
scan, compute speedup and Dice against pre-generated TS heart masks.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Dict, List, Tuple, cast

import shutil

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel import processing

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.predict_cac.segmentation.inference.predict import (  # noqa: E402
    load_model,
    predict_single_scan,
)
from src.predict_cac.utils.config_loader import load_config  # noqa: E402


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    return float((2 * inter) / (denom + 1e-8))


def merge_labels(label_files: List[pathlib.Path], ct_img: nib.Nifti1Image) -> nib.Nifti1Image:
    merged = np.zeros(ct_img.shape, dtype=np.uint8)
    for path in label_files:
        if not path.exists():
            print(f"Warning: missing label {path.name}")
            continue
        merged |= cast(nib.Nifti1Image, nib.load(str(path))).get_fdata().astype(np.uint8)
    merged = (merged > 0).astype(np.uint8)
    mask_img = nib.Nifti1Image(merged, ct_img.affine, ct_img.header)
    mask_img.set_data_dtype(np.uint8)
    return mask_img


def run_totalsegmentator(ct_path: pathlib.Path, output_dir: pathlib.Path, task: str, license_number: str) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "totalsegmentator",
        "-i",
        str(ct_path),
        "-o",
        str(output_dir),
        "-t",
        task,
        "--license_number",
        license_number,
        "--quiet",
    ]
    start = time.perf_counter()
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - start
    return elapsed


def load_split(data_dicts: List[Dict[str, str]], split_path: pathlib.Path) -> List[Dict[str, str]]:
    if split_path.exists():
        split_info = json.loads(split_path.read_text())
        return [d for d in data_dicts if split_info.get(pathlib.Path(d["image"]).stem) == "test"]
    return data_dicts


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model vs TotalSegmentator")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--license_number", required=True, help="TotalSegmentator license number")
    parser.add_argument("--max_scans", type=int, default=None, help="Optional cap on number of test scans to benchmark")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    outputs_cfg = config["outputs"]
    metrics_dir = pathlib.Path(outputs_cfg["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ts_cfg = config.get("totalsegmentator", {})
    ts_task = ts_cfg.get("benchmark_task") or ts_cfg.get("task") or "heartchambers_highres"
    if ts_task == "total":
        ts_task = "heartchambers_highres"
    labels_dir = pathlib.Path(outputs_cfg.get("labels_dir", ROOT / "outputs" / "labels"))

    from src.predict_cac.segmentation.datasets.heart_dataset import build_data_dicts  # noqa: E402

    data_dicts = build_data_dicts(data_cfg["image_dir"], data_cfg["label_dir"])
    test_list = load_split(data_dicts, metrics_dir / "data_split.json")
    if not test_list:
        print("No test data found; ensure data_split.json exists")
        sys.exit(1)

    model = load_model(args.checkpoint, config)

    # Warmup runs (clean up temp outputs after each)
    warmup_dir = metrics_dir / "ts_tmp" / "warmup"
    if warmup_dir.exists():
        shutil.rmtree(warmup_dir)
    for _ in range(config["benchmark"]["n_warmup_runs"]):
        warmup_dir.mkdir(parents=True, exist_ok=True)
        _ = run_totalsegmentator(pathlib.Path(test_list[0]["image"]), warmup_dir, ts_task, args.license_number)
        _ = predict_single_scan(model, test_list[0]["image"], config, test_list[0].get("label"))
        shutil.rmtree(warmup_dir, ignore_errors=True)

    if args.max_scans:
        test_list = test_list[: args.max_scans]

    records = []
    for entry in test_list:
        ct_path = pathlib.Path(entry["image"])
        scan_id = ct_path.stem
        ct_img = cast(nib.Nifti1Image, nib.load(str(ct_path)))

        ts_out_dir = metrics_dir / "ts_tmp" / scan_id
        if ts_out_dir.exists():
            shutil.rmtree(ts_out_dir)
        ts_out_dir.mkdir(parents=True, exist_ok=True)

        ts_time = run_totalsegmentator(ct_path, ts_out_dir, ts_task, args.license_number)
        ref_mask_path = labels_dir / f"{scan_id}_heart_mask.nii.gz"
        if ref_mask_path.exists():
            ts_mask_img = cast(nib.Nifti1Image, nib.load(str(ref_mask_path)))
            if ts_mask_img.shape != ct_img.shape:
                ts_mask_img = processing.resample_from_to(ts_mask_img, ct_img, order=0)
            ts_mask = ts_mask_img.get_fdata().astype(np.uint8)
            print(f"Loaded TS reference mask from {ref_mask_path}")
        else:
            print(f"Reference mask missing at {ref_mask_path}; falling back to TS outputs")
            available_files = sorted(p.name for p in ts_out_dir.glob("*.nii.gz"))
            print(f"TS outputs for {scan_id}: {available_files}")

            label_files: List[pathlib.Path] = []
            heart_file = ts_out_dir / "heart.nii.gz"
            if heart_file.exists():
                label_files.append(heart_file)

            substrings = ["heart", "atrium", "ventricle", "aorta", "pulmonary"]
            for path in ts_out_dir.glob("*.nii.gz"):
                if path in label_files:
                    continue
                stem = path.stem.lower()
                if any(s in stem for s in substrings):
                    label_files.append(path)

            if not label_files and available_files:
                print(f"No heart-related labels found for {scan_id}; merging all TS outputs instead.")
                label_files = list(ts_out_dir.glob("*.nii.gz"))

            ts_mask_img = merge_labels(label_files, ct_img)
            ts_mask = ts_mask_img.get_fdata().astype(np.uint8)

        pred_mask, model_time = predict_single_scan(model, str(ct_path), config, entry.get("label"))
        dice = dice_score(pred_mask, ts_mask)
        print(f"Mask voxels for {scan_id}: TS {int(ts_mask.sum())}, pred {int(pred_mask.sum())}")
        records.append(
            {
                "scan_id": scan_id,
                "ts_time_s": ts_time,
                "model_time_s": model_time,
                "speedup_x": ts_time / model_time if model_time > 0 else float("inf"),
                "dice": dice,
            }
        )
        print(f"{scan_id}: TS {ts_time:.2f}s | Model {model_time:.2f}s | Speedup {records[-1]['speedup_x']:.1f}x | Dice {dice:.3f}")

        shutil.rmtree(ts_out_dir, ignore_errors=True)

    df = pd.DataFrame(records)
    df.to_csv(metrics_dir / "benchmark_results.csv", index=False)
    summary = df.describe()
    print("\nSummary:\n", summary)


if __name__ == "__main__":
    main()
