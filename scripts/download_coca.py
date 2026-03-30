"""Download a subset of the COCA dataset from Azure Blob Storage.

Usage (defaults: 5 scans to data/coca_ct):
  python scripts/download_coca.py --output_dir data/coca_ct --max_scans 5
"""
from __future__ import annotations

import argparse
import pathlib
import tempfile
from typing import Iterable, List

from azure.storage.blob import ContainerClient
from tqdm import tqdm

SAS_URL = (
    "https://aimistanforddatasets01.blob.core.windows.net/"
    "cocacoronarycalciumandchestcts-2?sv=2019-02-02&sr=c&sig=YmFjX5PbybMXXEHQjpzSrpW5bwRhwiLrBRT1FcmmbuM%3D&st=2026-03-27T08%3A52%3A48Z&se=2026-04-26T08%3A57%3A48Z&sp=rl"
)


def _human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _download_blob(container: ContainerClient, blob_name: str, dest: pathlib.Path) -> int:
    blob = container.get_blob_client(blob_name)
    props = blob.get_blob_properties()
    size = int(props.size or 0)
    stream = blob.download_blob(max_concurrency=1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f, tqdm(
        total=size,
        unit="B",
        unit_scale=True,
        desc=dest.name,
        leave=False,
    ) as bar:
        for chunk in stream.chunks():
            f.write(chunk)
            bar.update(len(chunk))
    return size


def _convert_dicom_to_nifti(dicom_path: pathlib.Path, nifti_path: pathlib.Path) -> None:
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise SystemExit("SimpleITK is required for DICOM conversion. pip install SimpleITK") from exc

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(dicom_path))
    image = reader.Execute()
    sitk.WriteImage(image, str(nifti_path))


def _ordered_blob_names(container: ContainerClient, max_needed: int) -> Iterable[str]:
    # Prefer NIfTI blobs first, then anything else as fallback.
    nii: List[str] = []
    other: List[str] = []
    pager = container.list_blobs(results_per_page=500).by_page()
    try:
        for page in pager:
            for blob in page:
                name = blob.name
                if name.lower().endswith(".nii.gz"):
                    nii.append(name)
                else:
                    other.append(name)
            if len(nii) >= max_needed:
                break
    except Exception as exc:  # pragma: no cover - runtime logging
        print(f"Blob listing interrupted: {exc}")
    return list(nii) + other


def main() -> None:
    parser = argparse.ArgumentParser(description="Download COCA CT scans from Azure Blob Storage")
    parser.add_argument("--output_dir", type=pathlib.Path, default=pathlib.Path("data/coca_ct"))
    parser.add_argument("--max_scans", type=int, default=5)
    args = parser.parse_args()

    print("Connecting to Azure container...")
    container = ContainerClient.from_container_url(SAS_URL)
    print("Listing blobs...")
    blob_names = list(_ordered_blob_names(container, max_needed=args.max_scans * 10))
    if not blob_names:
        raise SystemExit("No blobs found at SAS URL; check access window or URL.")

    print(f"Preparing to attempt downloads; target count {args.max_scans}...")
    downloads = []
    attempted = 0
    skipped = 0
    for idx, blob_name in enumerate(blob_names, start=1):
        if len(downloads) >= args.max_scans:
            break
        target_name = pathlib.Path(blob_name).name
        target_path = args.output_dir / target_name

        # Handle potential DICOM by converting to patient_###.nii.gz
        attempted += 1
        try:
            if blob_name.lower().endswith(".nii.gz"):
                size = _download_blob(container, blob_name, target_path)
                downloads.append((target_path, size))
            elif blob_name.lower().endswith(('.dcm', '.dicom')):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = pathlib.Path(tmpdir) / target_name
                    size = _download_blob(container, blob_name, tmp_path)
                    nifti_path = args.output_dir / f"patient_{attempted:03d}.nii.gz"
                    _convert_dicom_to_nifti(tmp_path, nifti_path)
                    downloads.append((nifti_path, size))
            else:
                skipped += 1
                if skipped <= 5:
                    print(f"Skipping non-image blob: {blob_name}")
        except Exception as exc:  # pragma: no cover - runtime logging
            print(f"Failed to download {blob_name}: {exc}")
            continue

    if skipped > 5:
        print(f"Skipped {skipped} non-image blobs.")
    if len(downloads) < args.max_scans:
        print(
            f"Downloaded {len(downloads)} file(s), fewer than requested {args.max_scans}."
            " Some blobs were skipped or failed."
        )

    total_size = sum(sz for _, sz in downloads)
    print(f"Downloaded {len(downloads)} files -> {_human_size(total_size)}")
    for path, sz in downloads:
        print(f"  {path} ({_human_size(sz)})")


if __name__ == "__main__":
    main()
