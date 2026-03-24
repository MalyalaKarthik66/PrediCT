"""Device selection helpers for PyTorch/MONAI workloads."""

from __future__ import annotations

from typing import Any

import torch


def get_torch_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def torch_device_info(device: torch.device) -> dict[str, Any]:
    """Return a JSON-serializable summary of the selected compute device."""
    payload: dict[str, Any] = {
        "torch_version": torch.__version__,
        "selected_device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(idx)
        payload.update(
            {
                "cuda_device_index": idx,
                "cuda_device_name": props.name,
                "cuda_total_memory_bytes": int(props.total_memory),
            }
        )
    return payload
