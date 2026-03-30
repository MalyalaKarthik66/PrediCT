"""Run TotalSegmentator heartchambers_highres and merge 7 labels into a binary mask.

Docstring (copy to README/notebook): TotalSegmentator v2.x, task=heartchambers_highres,
labels merged: heart_myocardium, atrium_left, atrium_right, ventricle_left,
ventricle_right, aorta, pulmonary_artery. CLI example:
python scripts/generate_heart_labels.py --input_dir data/coca_ct --output_dir outputs/labels --license_number YOUR_KEY --config config.yaml
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile
from typing import List, cast

import nibabel as nib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.predict_cac.utils.config_loader import load_config  # noqa: E402


def merge_labels(label_files: List[pathlib.Path], ct_img: nib.Nifti1Image) -> nib.Nifti1Image:
    merged = np.zeros(ct_img.shape, dtype=np.uint8)
    for path in label_files:
        if not path.exists():
            print(f"Warning: missing label {path.name}")
            continue
        merged |= cast(nib.Nifti1Image, nib.load(str(path))).get_fdata().astype(np.uint8)
    merged = (merged > 0).astype(np.uint8)
    mask_img = nib.Nifti1Image(merged, ct_img.affine, ct_img.header)
    mask_img.set_data_dtype(np.uint8)
    return mask_img


def run_totalsegmentator(ct_path: pathlib.Path, output_dir: pathlib.Path, task: str, license_number: str) -> None:
    cmd = [
        "totalsegmentator",
        "-i",
        str(ct_path),
        "-o",
        str(output_dir),
        "-t",
        task,
        "--license_number",
        license_number,
        "--quiet",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate heart masks via TotalSegmentator")
    parser.add_argument("--input_dir", required=True, help="Directory of CT .nii.gz files")
    parser.add_argument("--output_dir", required=True, help="Where to save merged heart masks")
    parser.add_argument("--license_number", required=True, help="TotalSegmentator academic license key")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--max_scans", type=int, default=None, help="Optional limit for quick tests")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = config["totalsegmentator"]["labels_merged"]
    task = config["totalsegmentator"]["task"]

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ct_files = sorted(input_dir.glob("*.nii.gz"))
    if args.max_scans:
        ct_files = ct_files[: args.max_scans]

    success = 0
    failure = 0
    for ct_path in ct_files:
        mask_path = output_dir / f"{ct_path.stem}_heart_mask.nii.gz"
        if mask_path.exists():
            print(f"Skipping existing mask for {ct_path.name}")
            success += 1
            continue

        print(f"Processing {ct_path.name}...")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = pathlib.Path(tmpdir)
                run_totalsegmentator(ct_path, tmpdir_path, task, args.license_number)
                label_files = [tmpdir_path / f"{lbl}.nii.gz" for lbl in labels]
                ct_img = cast(nib.Nifti1Image, nib.load(str(ct_path)))
                merged_img = merge_labels(label_files, ct_img)
                nib.save(merged_img, mask_path)
                success += 1
        except subprocess.CalledProcessError as exc:
            print(f"TotalSegmentator failed for {ct_path.name}: {exc}")
            failure += 1
        except Exception as exc:  # noqa: BLE001
            print(f"Unexpected error for {ct_path.name}: {exc}")
            failure += 1

    total = len(ct_files)
    print(f"Completed: {success}/{total} successful, {failure} failed")


if __name__ == "__main__":
    main()
