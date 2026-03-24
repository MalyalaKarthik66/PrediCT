from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_script(root: Path, script_rel_path: str) -> None:
    script_path = root / script_rel_path
    venv_python = root / ".venv" / "Scripts" / "python.exe"
    python_exec = str(venv_python) if venv_python.exists() else sys.executable
    subprocess.run([python_exec, str(script_path)], cwd=str(root), check=True)


def test_pipeline_e2e() -> None:
    root = Path(__file__).resolve().parents[1]

    _run_script(root, "scripts/generate_demo_data.py")
    _run_script(root, "scripts/run_preprocessing.py")
    _run_script(root, "scripts/run_registration.py")
    _run_script(root, "scripts/run_validation.py")

    metrics_path = root / "outputs" / "validation_metrics.json"
    assert metrics_path.exists(), "outputs/validation_metrics.json should exist after e2e run"

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "percent_within_10mm" in payload, "Validation metrics must include percent_within_10mm"
