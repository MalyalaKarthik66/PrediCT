"""Inference helpers: load model, run single-scan/batch prediction, resample output."""
from __future__ import annotations

import pathlib
import time
from typing import Dict, List, Tuple, cast

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel import processing
from monai.inferers.utils import sliding_window_inference

from ..models.unet_3d import build_heart_unet
from ..utils.preprocessing import get_inference_transforms, get_val_transforms


def load_model(checkpoint_path: str | pathlib.Path, config: Dict) -> torch.nn.Module:
    model = build_heart_unet(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def predict_single_scan(
    model: torch.nn.Module,
    ct_path: str,
    config: Dict,
    label_path: str | None = None,
) -> Tuple[np.ndarray, float]:
    device = next(model.parameters()).device
    start = time.perf_counter()
    if label_path:
        transforms = get_val_transforms(config, crop=False)
        data = cast(Dict[str, object], transforms({"image": ct_path, "label": label_path}))
    else:
        transforms = get_inference_transforms(config)
        data = cast(Dict[str, object], transforms({"image": ct_path}))

    image_mt = cast(torch.Tensor, data["image"])
    image_meta = getattr(image_mt, "meta", None)
    affine_obj = image_meta.get("affine") if image_meta is not None else None
    affine = np.asarray(affine_obj) if affine_obj is not None else None
    base_tensor = image_mt.as_tensor() if hasattr(image_mt, "as_tensor") else image_mt
    image = torch.as_tensor(base_tensor).unsqueeze(0).float().to(device)  # add batch dim

    def _pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        spatial = x.shape[2:]
        target = [((d + multiple - 1) // multiple) * multiple for d in spatial]
        pad = [t - s for t, s in zip(target, spatial)]
        if all(p == 0 for p in pad):
            return x, (0, 0, 0)
        pad_args = (0, pad[2], 0, pad[1], 0, pad[0])  # W, H, D padding on the "right"
        return F.pad(x, pad_args), (pad[0], pad[1], pad[2])

    roi_size = tuple(config["training"]["roi_size"])
    sw_batch_size = int(config.get("benchmark", {}).get("batch_size", 1))
    image_padded, pad_amounts = _pad_to_multiple(image)

    start = time.perf_counter()
    with torch.no_grad():
        use_amp = device.type == "cuda" and bool(config["training"].get("mixed_precision", True))
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            spatial = image_padded.shape[2:]
            if all(s <= r for s, r in zip(spatial, roi_size)):
                # Small volumes: run a single forward pass on the padded tensor
                logits = model(image_padded)
            else:
                logits = sliding_window_inference(
                    image_padded,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                )
        logits = cast(torch.Tensor, logits)
        if any(pad_amounts):
            d_pad, h_pad, w_pad = pad_amounts
            logits = logits[
                :,
                :,
                : logits.shape[2] - d_pad if d_pad else logits.shape[2],
                : logits.shape[3] - h_pad if h_pad else logits.shape[3],
                : logits.shape[4] - w_pad if w_pad else logits.shape[4],
            ]
        pred_tensor = torch.argmax(logits, dim=1).float()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    pred_np = pred_tensor.cpu().numpy()[0].astype(np.uint8)
    if affine is None:
        affine = np.eye(4)

    pred_img = nib.Nifti1Image(pred_np, affine)
    orig_img = cast(nib.Nifti1Image, nib.load(ct_path))
    resampled_img = processing.resample_from_to(pred_img, orig_img, order=0)
    pred_resampled = resampled_img.get_fdata().astype(np.uint8)
    elapsed = time.perf_counter() - start
    return pred_resampled, elapsed


def predict_batch(
    model: torch.nn.Module,
    test_data_dicts: List[Dict[str, str]],
    config: Dict,
    output_dir: str | pathlib.Path,
) -> List[Dict[str, object]]:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []
    for entry in test_data_dicts:
        ct_path = entry["image"]
        scan_id = pathlib.Path(ct_path).stem
        pred_resampled, inference_time = predict_single_scan(model, ct_path, config)
        orig_img = cast(nib.Nifti1Image, nib.load(ct_path))
        save_img = nib.Nifti1Image(pred_resampled.astype(np.uint8), orig_img.affine)
        save_path = output_dir / f"{scan_id}_pred_mask.nii.gz"
        nib.save(save_img, save_path)
        results.append({"scan_id": scan_id, "pred_path": str(save_path), "inference_time_s": inference_time})
    return results
