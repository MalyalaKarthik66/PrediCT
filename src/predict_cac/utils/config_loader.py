"""Configuration loader and validator for the PrediCT heart segmentation pipeline.

This module centralizes YAML parsing so every script pulls settings from a single
source of truth (config.yaml), as required by the project blueprint. Validation
checks required sections/keys early to fail fast with clear messages.
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict, Iterable

import yaml


_REQUIRED_KEYS = {
    "data": ["image_dir", "label_dir", "spacing", "hu_min", "hu_max", "train_frac", "val_frac", "test_frac"],
    "model": ["spatial_dims", "in_channels", "out_channels", "channels", "strides", "num_res_units", "dropout"],
    "training": ["epochs", "lr", "batch_size", "seed", "mixed_precision", "early_stopping_patience", "roi_size"],
    "totalsegmentator": ["task", "version", "labels_merged"],
    "label_qc": ["volume_min_mL", "volume_max_mL"],
    "benchmark": ["gpu_name", "batch_size", "n_warmup_runs", "timing_scope"],
    "outputs": ["checkpoint_dir", "metrics_dir", "figures_dir", "labels_dir", "label_qc_dir"],
}


def _assert_keys(section: str, config_section: Dict[str, Any], required: Iterable[str]) -> None:
    missing = [k for k in required if k not in config_section]
    if missing:
        raise ValueError(f"Missing keys in section '{section}': {', '.join(missing)}")


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate required sections and keys, returning the same dict if valid.

    Raises:
        ValueError: if any required section/key is missing or fractions are invalid.
    """
    for section, keys in _REQUIRED_KEYS.items():
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config")
        _assert_keys(section, config[section], keys)

    fractions = (
        float(config["data"].get("train_frac", 0)),
        float(config["data"].get("val_frac", 0)),
        float(config["data"].get("test_frac", 0)),
    )
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("Train/val/test fractions must sum to 1.0")
    return config


def load_config(config_path: str | pathlib.Path = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file and validate required fields."""
    path = pathlib.Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return validate_config(config)
