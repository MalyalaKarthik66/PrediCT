"""CLI to run atlas registration for a moving/fixed scan pair."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict_cac.registration.atlas_registration import run_registration


def main() -> None:
    parser = argparse.ArgumentParser(description="Run atlas registration")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "registration" / "affine.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    strategy = "rigid_affine"
    moving_path = ROOT / cfg["moving_image"]
    fixed_path = ROOT / cfg["fixed_image"]
    out_dir = ROOT / cfg.get("output_dir", "outputs/registered_atlas")

    if not moving_path.exists() or not fixed_path.exists():
        result = {
            "status": "skipped",
            "reason": "Missing moving/fixed image path",
            "moving_image": str(moving_path),
            "fixed_image": str(fixed_path),
            "output_dir": str(out_dir),
        }
        print(json.dumps(result, indent=2))
        return

    start = time.time()
    result = run_registration(
        moving_image_path=moving_path,
        fixed_image_path=fixed_path,
        out_dir=out_dir,
        strategy=strategy,
        levels=int(cfg.get("levels", 3)),
        iterations=[int(v) for v in cfg.get("iterations", [1000, 500, 200])],
        shrink_factors=[int(v) for v in cfg.get("shrink_factors", [4, 2, 1])],
        smoothing_sigmas=[float(v) for v in cfg.get("smoothing_sigmas", [2, 1, 0])],
        seed=seed,
    )
    runtime_seconds = time.time() - start
    result["runtime_seconds"] = float(round(runtime_seconds, 4))

    metrics_path = ROOT / "experiments" / "metrics.json"
    metrics_payload: dict[str, object] = {}
    if metrics_path.exists():
        try:
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics_payload = {}

    metrics_payload.update(
        {
            "registration_strategy": strategy,
            "runtime_seconds": result["runtime_seconds"],
        }
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
