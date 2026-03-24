"""CLI to compute centerline-proximity validation metric."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict_cac.evaluation.registration_eval import evaluate_registration, save_metrics_json
from predict_cac.evaluation.visualization import save_registration_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation metric")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "segmentation.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    np.random.seed(int(cfg.get("seed", 42)))
    calcium_path = ROOT / cfg["calcium_mask"]
    centerline_path = ROOT / cfg["centerline_mask"]
    image_path = ROOT / cfg.get("ct_image", cfg.get("calcium_mask", ""))
    threshold_mm = float(cfg.get("threshold_mm", 10.0))

    if calcium_path.exists() and centerline_path.exists():
        result = evaluate_registration(
            calcium_mask_path=calcium_path,
            centerline_mask_path=centerline_path,
            threshold_mm=threshold_mm,
        )
        if image_path.exists():
            result["plots"] = save_registration_plots(
                image_path=image_path,
                calcium_mask_path=calcium_path,
                centerline_mask_path=centerline_path,
                out_dir=ROOT / "outputs" / "plots",
            )
    else:
        result = {
            "status": "skipped",
            "reason": "Missing calcium or centerline mask path",
            "calcium_mask": str(calcium_path),
            "centerline_mask": str(centerline_path),
            "threshold_mm": threshold_mm,
            "mean_distance_mm": 0.0,
            "median_distance_mm": 0.0,
            "percent_within_10mm": 0.0,
        }

    metrics_path = ROOT / "experiments" / "metrics.json"
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                for key in ("runtime_seconds", "registration_strategy"):
                    if key in existing:
                        result[key] = existing[key]
        except Exception:
            pass

    save_metrics_json(result, metrics_path)

    (ROOT / "outputs" / "validation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
