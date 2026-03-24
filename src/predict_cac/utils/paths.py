"""Centralized project path resolution for the PrediCT CAC prototype."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return repository root from current file location."""
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    return project_root() / "data"


def outputs_dir() -> Path:
    return project_root() / "outputs"


def experiments_dir() -> Path:
    return project_root() / "experiments"


def configs_dir() -> Path:
    return project_root() / "configs"
