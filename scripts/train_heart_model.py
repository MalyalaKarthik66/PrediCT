"""Train lightweight 3D U-Net for heart segmentation (per blueprint)."""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.predict_cac.segmentation.datasets.heart_dataset import (  # noqa: E402
    build_data_dicts,
    compute_dataset_statistics,
    get_dataloaders,
    patient_level_split,
)
from src.predict_cac.segmentation.models.unet_3d import build_heart_unet  # noqa: E402
from src.predict_cac.segmentation.training.train import train  # noqa: E402
from src.predict_cac.segmentation.utils.preprocessing import (  # noqa: E402
    get_train_transforms,
    get_val_transforms,
)
from src.predict_cac.utils.config_loader import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the heart segmentation model")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    outputs_cfg = config["outputs"]
    metrics_dir = pathlib.Path(outputs_cfg["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    data_dicts = build_data_dicts(data_cfg["image_dir"], data_cfg["label_dir"])
    split_path = metrics_dir / "data_split.json"
    train_list, val_list, test_list = patient_level_split(
        data_dicts,
        data_cfg["train_frac"],
        data_cfg["val_frac"],
        seed=config["training"]["seed"],
        split_path=split_path,
    )
    if not train_list or not val_list:
        print("Train/val splits are empty; check data paths and labels")
        sys.exit(1)

    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)
    train_loader, val_loader = get_dataloaders(
        train_list,
        val_list,
        train_transforms,
        val_transforms,
        batch_size=config["training"]["batch_size"],
    )

    model = build_heart_unet(config)
    history = train(model, train_loader, val_loader, config)
    best_dice = max(history.get("val_dice", [0.0])) if history else 0.0

    print(f"Training complete. Best Val Dice: {best_dice:.4f}")
    print(f"Model saved to {outputs_cfg['checkpoint_dir']}/best_model.pth")

    stats_df = compute_dataset_statistics(train_list + val_list + test_list)
    stats_df.to_csv(metrics_dir / "dataset_stats.csv", index=False)


if __name__ == "__main__":
    main()
