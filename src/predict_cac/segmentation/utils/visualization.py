"""Visualization helpers for training curves, metric distributions, overlays, and CAC demo."""
from __future__ import annotations

import json
import pathlib
from typing import Tuple, cast

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from nibabel import processing

sns.set_style("whitegrid")


def _ensure_dir(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_training_curves(history_json_path: str | pathlib.Path, save_path: str | pathlib.Path):
    with open(history_json_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    train_loss = history.get("train_loss", [])
    val_dice = history.get("val_dice", [])
    epochs = np.arange(1, len(train_loss) + 1)
    best_epoch = int(np.argmax(val_dice) + 1) if val_dice else 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="Train Loss")
    if best_epoch:
        axes[0].axvline(best_epoch, linestyle="--", color="gray", label="Best Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, val_dice, label="Val Dice", color="tab:green")
    if best_epoch:
        axes[1].axvline(best_epoch, linestyle="--", color="gray", label="Best Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()

    plt.tight_layout()
    save_path = pathlib.Path(save_path)
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return fig


def plot_dice_distribution(per_case_df, save_path: str | pathlib.Path, target_dice: float = 0.85):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=per_case_df, y="dice", ax=ax, color="#c7dfff")
    sns.swarmplot(data=per_case_df, y="dice", ax=ax, color="#1f78b4", size=4)
    ax.axhline(target_dice, linestyle="--", color="red", label=f"Target {target_dice}")
    mean = per_case_df["dice"].mean()
    std = per_case_df["dice"].std()
    ax.text(0.05, 0.95, f"mean={mean:.3f}\nstd={std:.3f}", transform=ax.transAxes, va="top")
    ax.set_ylabel("Dice Score")
    ax.set_title("Test Set Dice Distribution")
    ax.legend()

    save_path = pathlib.Path(save_path)
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def _get_center_slices(mask: np.ndarray) -> Tuple[int, int, int]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        z = mask.shape[2] // 2
        y = mask.shape[1] // 2
        x = mask.shape[0] // 2
    else:
        z = int(np.median(coords[:, 2]))
        y = int(np.median(coords[:, 1]))
        x = int(np.median(coords[:, 0]))
    return x, y, z


def _overlay_slice(ct_slice: np.ndarray, gt_slice: np.ndarray, pred_slice: np.ndarray, ax, title: str):
    ax.imshow(ct_slice, cmap="gray")
    tp = np.logical_and(gt_slice, pred_slice)
    fp = np.logical_and(pred_slice, np.logical_not(gt_slice))
    fn = np.logical_and(gt_slice, np.logical_not(pred_slice))
    overlay = np.zeros((*ct_slice.shape, 3), dtype=float)
    overlay[..., 0] = fp  # red
    overlay[..., 1] = tp  # green
    overlay[..., 2] = fn  # blue
    ax.imshow(overlay, alpha=0.4)
    ax.set_axis_off()
    ax.set_title(title)


def plot_segmentation_overlay(
    ct_path: str,
    gt_mask_path: str,
    pred_mask_path: str,
    scan_id: str,
    dice: float,
    save_path: str | pathlib.Path,
):
    ct_img = cast(nib.Nifti1Image, nib.load(ct_path))
    gt_img = cast(nib.Nifti1Image, nib.load(gt_mask_path))
    pred_img = cast(nib.Nifti1Image, nib.load(pred_mask_path))
    ct = ct_img.get_fdata()
    gt = gt_img.get_fdata().astype(bool)
    pred = pred_img.get_fdata().astype(bool)
    x, y, z = _get_center_slices(gt)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    _overlay_slice(ct[:, :, z], gt[:, :, z], pred[:, :, z], axes[0], "Axial")
    _overlay_slice(ct[:, y, :], gt[:, y, :], pred[:, y, :], axes[1], "Coronal")
    _overlay_slice(ct[x, :, :], gt[x, :, :], pred[x, :, :], axes[2], "Sagittal")
    plt.suptitle(f"{scan_id} | Dice: {dice:.4f}")

    save_path = pathlib.Path(save_path)
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_cac_masking_demo(
    raw_ct_path: str,
    heart_mask_path: str,
    scan_id: str,
    save_path: str | pathlib.Path,
) -> float:
    raw_ct_img = cast(nib.Nifti1Image, nib.load(raw_ct_path))
    raw_ct = raw_ct_img.get_fdata()
    mask_img = cast(nib.Nifti1Image, nib.load(heart_mask_path))
    if mask_img.shape != raw_ct_img.shape:
        mask_img = processing.resample_from_to(mask_img, raw_ct_img, order=0)
    mask = mask_img.get_fdata().astype(bool)

    cac_before = raw_ct > 130
    cac_after = np.logical_and(cac_before, mask)
    n_before = float(cac_before.sum())
    n_after = float(cac_after.sum())
    reduction_pct = 100.0 * (1.0 - n_after / (n_before + 1e-8))

    # Choose axial slice through heart centroid for visualization
    x, y, z = _get_center_slices(mask.astype(np.uint8))
    slices = [
        raw_ct[:, :, z],
        mask[:, :, z].astype(float),
        cac_before[:, :, z].astype(float),
        cac_after[:, :, z].astype(float),
    ]
    titles = [
        "Original CT (HU)",
        "Predicted Heart Mask",
        "HU > 130 (before)",
        "HU > 130 (after)",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, img, title in zip(axes, slices, titles):
        ax.imshow(img, cmap="gray")
        ax.set_axis_off()
        ax.set_title(title)
    plt.suptitle(f"{scan_id} | FP Reduction: {reduction_pct:.1f}%")

    save_path = pathlib.Path(save_path)
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return reduction_pct
