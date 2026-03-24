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
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    np.random.seed(int(cfg.get("seed", 42)))
    input_dir = ROOT / cfg["input_nifti_dir"]
    output_dir = ROOT / cfg["output_nifti_dir"]
    spacing = tuple(cfg.get("target_spacing", [1.0, 1.0, 1.0]))

    outputs = run_preprocessing_pipeline(input_nifti_dir=input_dir, output_dir=output_dir, target_spacing=spacing)
    print(f"Preprocessed {len(outputs)} scans")


if __name__ == "__main__":
    main()
