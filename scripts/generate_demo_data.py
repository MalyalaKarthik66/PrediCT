"""Generate synthetic demo data for end-to-end CAC pipeline execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict_cac.utils.synthetic_data import generate_synthetic_demo_data


def main() -> None:
    np.random.seed(42)
    paths = generate_synthetic_demo_data(ROOT)
    print(
        json.dumps(
            {
                "fixed_nifti": str(paths.fixed_nifti),
                "moving_nifti": str(paths.moving_nifti),
                "fixed_preprocessed": str(paths.fixed_preprocessed),
                "moving_preprocessed": str(paths.moving_preprocessed),
                "calcium_mask": str(paths.calcium_mask),
                "centerline_mask": str(paths.centerline_mask),
                "moving_centerline_mask": str(paths.moving_centerline_mask),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
