"""Minimal training harness placeholder for CAC segmentation experiments."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import monai
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict_cac.datasets.coca_dataset import create_train_val_test_splits
from predict_cac.datasets.dataloader import build_segmentation_dataloaders
from predict_cac.utils.device import get_torch_device, torch_device_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CAC segmentation model")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "segmentation.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    device = get_torch_device(prefer_cuda=bool(cfg.get("prefer_cuda", True)))

    data_cfg = cfg.get("data", {})
    split_cfg = cfg.get("split", {})
    loader_cfg = cfg.get("loader", {})
    train_cfg = cfg.get("training", {})

    image_dir = ROOT / data_cfg.get("image_dir", "data/preprocessed")
    mask_dir = ROOT / data_cfg.get("mask_dir", "data/masks")
    metadata_csv = ROOT / data_cfg.get("metadata_csv", "data/metadata.csv")

    split_preview = create_train_val_test_splits(
        image_dir=image_dir,
        mask_dir=mask_dir,
        metadata_csv=metadata_csv,
        seed=int(split_cfg.get("seed", 42)),
        train_ratio=float(split_cfg.get("train", 0.70)),
        val_ratio=float(split_cfg.get("val", 0.15)),
        test_ratio=float(split_cfg.get("test", 0.15)),
    )

    if not split_preview["train"]:
        raise RuntimeError(
            "No train samples discovered. Ensure preprocessed images exist in data.image_dir "
            "and masks exist in data.mask_dir with matching filenames."
        )

    dataloaders = build_segmentation_dataloaders(
        {
            **cfg,
            "data": {
                **data_cfg,
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir),
                "metadata_csv": str(metadata_csv),
            },
        }
    )

    max_train_batches = int(train_cfg.get("max_train_batches", 3))
    train_batch_shapes: list[dict[str, object]] = []
    for batch_index, batch in enumerate(dataloaders["train"]):
        if batch_index >= max_train_batches:
            break
        image = batch["image"]
        mask = batch["mask"]
        train_batch_shapes.append(
            {
                "batch": batch_index,
                "image_shape": list(image.shape),
                "mask_shape": list(mask.shape),
                "mask_positive_fraction": float((mask > 0).float().mean().item()),
            }
        )

    summary = {
        "status": "pipeline_validated",
        "model_name": cfg.get("model_name", "unet"),
        "epochs": int(cfg.get("epochs", 1)),
        "torch_version": torch.__version__,
        "monai_version": monai.__version__,
        "device": torch_device_info(device),
        "split_counts": {k: len(v) for k, v in split_preview.items()},
        "loader_batch_size": int(loader_cfg.get("batch_size", 1)),
        "loader_num_workers": int(loader_cfg.get("num_workers", 0)),
        "patch_size": loader_cfg.get("patch_size", [64, 64, 32]),
        "positive_patch_probability": float(loader_cfg.get("positive_patch_probability", 0.8)),
        "train_batches_preview": train_batch_shapes,
        "simpleitk_note": "SimpleITK registration/preprocessing runs on CPU.",
        "note": "Data split, CacheDataset loading, patch sampling, and train-time augmentation validated.",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
