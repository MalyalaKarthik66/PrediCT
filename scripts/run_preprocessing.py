"""CLI to run CT preprocessing pipeline for CAC segmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict_cac.transforms.preprocessing_pipeline import run_preprocessing_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "preprocessing.yaml")
    parser.add_argument("--max-scans", type=int, default=None, help="Optionally limit processed scans for quick validation")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    np.random.seed(int(cfg.get("seed", 42)))
    input_dir = ROOT / cfg["input_nifti_dir"]
    output_dir = ROOT / cfg["output_nifti_dir"]
    spacing = tuple(cfg.get("target_spacing", [1.0, 1.0, 1.0]))
    hu_min = float(cfg.get("hu_min", -200.0))
    hu_max = float(cfg.get("hu_max", 1000.0))
    roi_margin = int(cfg.get("roi_margin", 8))
    normalization_mode = str(cfg.get("normalization_mode", "per_scan"))
    apply_augmentation = bool(cfg.get("augmentation", {}).get("enabled", False))
    max_scans = args.max_scans if args.max_scans is not None else cfg.get("max_scans")

    outputs = run_preprocessing_pipeline(
        input_nifti_dir=input_dir,
        output_dir=output_dir,
        target_spacing=spacing,
        hu_min=hu_min,
        hu_max=hu_max,
        normalization_mode=normalization_mode,
        roi_margin=roi_margin,
        apply_augmentation=apply_augmentation,
        max_scans=max_scans,
    )
    print(f"Preprocessed {len(outputs)} scans")


if __name__ == "__main__":
    main()
