"""CLI for ingesting COCA DICOM scans into NIfTI format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict_cac.data.ingest_coca import run_ingest
from predict_cac.data.metadata_builder import build_metadata_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COCA DICOM ingestion")
    parser.add_argument("--dicom-root", type=Path, default=ROOT / "data" / "raw" / "dicom")
    parser.add_argument("--nifti-root", type=Path, default=ROOT / "data" / "nifti")
    parser.add_argument("--metadata-csv", type=Path, default=ROOT / "data" / "metadata.csv")
    args = parser.parse_args()

    df = run_ingest(dicom_root=args.dicom_root, nifti_root=args.nifti_root)
    metadata_df = build_metadata_csv(args.nifti_root, args.metadata_csv)
    print(f"Ingested {len(df)} scans into {args.nifti_root}")
    print(f"Metadata rows: {len(metadata_df)} written to {args.metadata_csv}")


if __name__ == "__main__":
    main()
