"""Training loop for the COCA heart segmentation model."""
from __future__ import annotations

import json
import pathlib
import random
from typing import Dict, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from monai.data.dataloader import DataLoader
from monai.inferers.utils import sliding_window_inference
from monai.metrics.meandice import DiceMetric
from monai.transforms.compose import Compose

from .loss import get_loss_function


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _prepare_dirs(outputs_cfg: Dict[str, str]) -> Tuple[pathlib.Path, pathlib.Path]:
    ckpt_dir = pathlib.Path(outputs_cfg["checkpoint_dir"])
    metrics_dir = pathlib.Path(outputs_cfg.get("metrics_dir", ckpt_dir))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir, metrics_dir


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
) -> Dict[str, list[float]]:
    training_cfg = config["training"]
    outputs_cfg = config["outputs"]
    set_seed(training_cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = get_loss_function(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg["lr"])
    use_amp = device.type == "cuda" and bool(training_cfg.get("mixed_precision", True))
    scaler = torch.amp.GradScaler("cuda") if use_amp else torch.amp.GradScaler(enabled=False)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 16) -> torch.Tensor:
        spatial = tensor.shape[2:]
        target = [((dim + multiple - 1) // multiple) * multiple for dim in spatial]
        pad_amounts = [t - s for t, s in zip(target, spatial)]
        if all(p == 0 for p in pad_amounts):
            return tensor
        # Pad order for F.pad: (W_left, W_right, H_left, H_right, D_left, D_right)
        pad = (0, pad_amounts[2], 0, pad_amounts[1], 0, pad_amounts[0])
        return F.pad(tensor, pad)

    def _match_spatial_dims(preds: torch.Tensor, lbls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Crop both tensors to the minimum shared D/H/W so loss shapes align
        min_shape = [min(p, l) for p, l in zip(preds.shape[2:], lbls.shape[2:])]
        preds = preds[:, :, : min_shape[0], : min_shape[1], : min_shape[2]]
        lbls = lbls[:, :, : min_shape[0], : min_shape[1], : min_shape[2]]
        return preds, lbls

    def _one_hot(labels: torch.Tensor) -> torch.Tensor:
        labels = labels.squeeze(1).long()
        oh = torch.nn.functional.one_hot(labels, num_classes=2)
        return oh.permute(0, 4, 1, 2, 3).float()

    ckpt_dir, metrics_dir = _prepare_dirs(outputs_cfg)
    history: Dict[str, list[float]] = {"train_loss": [], "val_dice": []}
    best_dice = -1.0
    best_epoch = 0
    patience = int(training_cfg.get("early_stopping_patience", 5))
    epochs_no_improve = 0
    roi_size = training_cfg["roi_size"]
    max_epochs = int(training_cfg.get("max_epochs", training_cfg.get("epochs", 0)))

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            inputs = batch["image"].to(device)
            labels = (batch["label"].to(device) > 0.5).long()
            padded_inputs = _pad_to_multiple(inputs)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(padded_inputs)
                outputs, labels_aligned = _match_spatial_dims(outputs, labels)
                loss = loss_fn(outputs, labels_aligned)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(len(train_loader), 1)

        model.eval()
        with torch.no_grad():
            val_pred_vox = 0
            val_label_vox = 0
            for batch in val_loader:
                val_inputs = batch["image"].to(device)
                val_labels = (batch["label"].to(device) > 0.5).long()
                padded_val_inputs = _pad_to_multiple(val_inputs)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    val_outputs = sliding_window_inference(
                        padded_val_inputs,
                        roi_size=roi_size,
                        sw_batch_size=1,
                        predictor=model,
                    )
                    val_outputs, val_labels = _match_spatial_dims(val_outputs, val_labels)
                pred_class = val_outputs.argmax(dim=1, keepdim=True)
                val_outputs_t = _one_hot(pred_class)
                val_labels_t = _one_hot(val_labels)
                val_pred_vox += int(pred_class.sum().item())
                val_label_vox += int(val_labels.sum().item())
                dice_metric(y_pred=[val_outputs_t], y=[val_labels_t])

            aggregated = dice_metric.aggregate()
            val_dice = float(cast(torch.Tensor, aggregated).item()) if dice_metric.get_buffer() is not None else 0.0
            dice_metric.reset()

        history["train_loss"].append(avg_train_loss)
        history["val_dice"].append(val_dice)
        print(
            f"Epoch {epoch}/{max_epochs} | Loss: {avg_train_loss:.4f} | Val Dice: {val_dice:.4f} | "
            f"Val fg vox: pred {val_pred_vox} / label {val_label_vox}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}; best Dice {best_dice:.4f} at epoch {best_epoch}")
            break

    torch.save(model.state_dict(), ckpt_dir / "last_model.pth")

    history_path = metrics_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history
