"""Minimal training harness placeholder for CAC segmentation experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import monai
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict_cac.utils.device import get_torch_device, torch_device_info


def main() -> None:
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="Train CAC segmentation model")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "segmentation.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = get_torch_device(prefer_cuda=bool(cfg.get("prefer_cuda", True)))
    summary = {
        "status": "initialized",
        "model_name": cfg.get("model_name", "unet"),
        "epochs": int(cfg.get("epochs", 1)),
        "torch_version": torch.__version__,
        "monai_version": monai.__version__,
        "device": torch_device_info(device),
        "simpleitk_note": "SimpleITK registration/preprocessing runs on CPU.",
        "note": "Plug in MONAI training loop in this script for full experiments.",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
