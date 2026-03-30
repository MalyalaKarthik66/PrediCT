"""Volume-based QC for merged heart masks (flags <50 mL or >600 mL)."""
from __future__ import annotations

import argparse
import pathlib
import sys

import nibabel as nib
import pandas as pd
from typing import cast, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.predict_cac.utils.config_loader import load_config  # noqa: E402


def compute_volume_ml(mask_path: pathlib.Path) -> float:
    img = cast(nib.Nifti1Image, nib.load(str(mask_path)))
    spacing = cast(Tuple[float, float, float], img.header.get_zooms()[:3])
    voxel_vol = float(spacing[0]) * float(spacing[1]) * float(spacing[2])
    voxels = img.get_fdata()
    heart_voxels = int((voxels > 0).sum())
    return float(heart_voxels * voxel_vol / 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="QC TotalSegmentator heart masks by volume")
    parser.add_argument("--label_dir", required=True, help="Directory of *_heart_mask.nii.gz files")
    parser.add_argument("--output_dir", required=True, help="Where to save label_volume_stats.csv")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    vol_min = config["label_qc"]["volume_min_mL"]
    vol_max = config["label_qc"]["volume_max_mL"]

    label_dir = pathlib.Path(args.label_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for mask_path in sorted(label_dir.glob("*_heart_mask.nii.gz")):
        volume_ml = compute_volume_ml(mask_path)
        flagged = volume_ml < vol_min or volume_ml > vol_max
        records.append({"scan_id": mask_path.stem, "volume_mL": volume_ml, "flagged": flagged})

    df = pd.DataFrame(records)
    if df.empty:
        print("No masks found to verify")
        sys.exit(1)

    csv_path = output_dir / "label_volume_stats.csv"
    df.to_csv(csv_path, index=False)
    print("Volume statistics:\n", df["volume_mL"].describe())
    flagged_df = df[df["flagged"]]
    if not flagged_df.empty:
        print("Flagged scans (inspect visually):")
        print(flagged_df[["scan_id", "volume_mL"]])
        sys.exit(1)
    else:
        print("All masks within acceptable volume range")


if __name__ == "__main__":
    main()
