from __future__ import annotations

from azure.storage.blob import ContainerClient
from pathlib import Path
import SimpleITK as sitk
import tempfile
import sys

SAS_URL = "https://aimistanforddatasets01.blob.core.windows.net/cocacoronarycalciumandchestcts-2?sv=2019-02-02&sr=c&sig=YmFjX5PbybMXXEHQjpzSrpW5bwRhwiLrBRT1FcmmbuM%3D&st=2026-03-27T08%3A52%3A48Z&se=2026-04-26T08%3A57%3A48Z&sp=rl"
MAX_PATIENTS = 70
OUT_DIR = Path("data/coca_ct")
OUT_DIR.mkdir(parents=True, exist_ok=True)

container = ContainerClient.from_container_url(SAS_URL)

for patient_id in range(MAX_PATIENTS):
    out_path = OUT_DIR / f"patient_{patient_id:03d}.nii.gz"
    if out_path.exists():
        print(f"Patient {patient_id}: already exists, skipping")
        continue

    prefix = f"Gated_release_final/patient/{patient_id}/"
    blobs = list(container.list_blobs(name_starts_with=prefix))
    if not blobs:
        print(f"Patient {patient_id}: no blobs found, skipping")
        continue

    print(f"Patient {patient_id}: downloading {len(blobs)} DICOM slices...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for blob in blobs:
                fname = Path(blob.name).name
                dest = tmp_path / fname
                client = container.get_blob_client(blob.name)
                with open(dest, "wb") as f:
                    f.write(client.download_blob(max_concurrency=1).readall())

            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(str(tmp_path))
            reader.SetFileNames(dicom_files)
            image = reader.Execute()
            sitk.WriteImage(image, str(out_path))
            size_mb = out_path.stat().st_size / 1e6
            print(f"  Saved: {out_path.name} ({size_mb:.1f} MB)")
    except Exception as exc:
        print(f"Patient {patient_id}: failed with {exc}")
        sys.exit(1)

print("Done downloading up to 50 patients.")
