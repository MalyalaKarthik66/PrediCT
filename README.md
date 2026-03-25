# PrediCT CAC Prototype

Building and comparing segmentation strategies for Coronary Artery Calcium (CAC) in the ML4SCI PrediCT project.

This repository now includes a modular, research-grade pipeline that extends existing COCA scripts and adds a clean package structure for ingestion, preprocessing, atlas registration, validation metrics, and Agatston scoring.

## Pipeline Diagram

```mermaid
flowchart LR
	A[DICOM] --> B[NIfTI]
	B --> C[Preprocessing]
	C --> D[Atlas Registration]
	D --> E[Centerline Transform]
	E --> F[Distance Metrics]
	F --> G[Visualization]
```

## Simple Architecture Diagram

```mermaid
flowchart LR
	A[DICOM] --> B[Preprocessing]
	B --> C[Registration]
	C --> D[Evaluation]
	D --> E[Visualization]
```

## Pipeline Overview

1. Ingest COCA DICOM scans and convert them to NIfTI (`data/nifti/`).
2. Build metadata (patient ID, slice thickness, spacing, scan size).
3. Preprocess CT volumes:
   - HU clipping (`-200` to `1000`)
   - z-score normalization
   - isotropic resampling (`1mm`)
   - cardiac ROI cropping
   - optional augmentation hook
4. Register atlas/centerlines to patient scans using **rigid initialization followed by affine refinement, with a lightweight retry mechanism for failed cases**. The retry mechanism automatically re-registers scans with optimized parameters if the initial attempt yields poor alignment quality (<30% alignment).
5. Compute validation metrics: mean/median distance and percentage of calcium voxels within `10mm` of transformed centerlines.
6. Save run metrics to `experiments/registration_comparison.csv` and `experiments/registration_comparison_summary.csv`, and documentation figures to `docs/images/`.

## Repository Layout

```text
data/
outputs/
src/predict_cac/
scripts/
configs/
notebooks/
experiments/
tests/
```

## Environment Setup

### 1) Create virtual environment

```bash
python -m venv .venv
```

### 2) Activate virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

Conda alternative:

```bash
conda env create -f environment.yml
conda activate predict-cac
```

## Dataset Preparation

1. Place the COCA dataset DICOM folders under `data/raw/dicom/coca/`.
2. Place ImageCAS coronary atlas resources under `data/atlas/imagecas/`.
3. (Optional) Place calcium masks under `data/masks/` and transformed centerline masks under `outputs/centerlines_warped/` for validation.
4. Update sample paths in:
	- `configs/preprocessing.yaml`
	- `configs/registration/*.yaml`
	- `configs/segmentation.yaml`

## Run the End-to-End Pipeline

```bash
python scripts/run_ingest.py
python scripts/run_preprocessing.py
python scripts/run_registration.py
python scripts/run_validation.py
```

## Using Real Datasets (COCA + ImageCAS)

Expected folder layout:

```text
data/
	raw/
		dicom/
			coca/
	atlas/
		imagecas/
```

Typical command sequence for real-scan processing:

```bash
python scripts/run_ingest.py
python scripts/run_preprocessing.py
python scripts/run_registration.py --config configs/registration/affine.yaml
python scripts/run_validation.py --config configs/segmentation.yaml
```

Update YAML paths in `configs/` to point to your real scan files, masks, and atlas-derived centerlines.

## Running the Demo Pipeline

Use this mode when COCA data is not available. It generates synthetic CT volumes,
calcium mask, and centerline mask so registration and validation produce real outputs.

```bash
python scripts/generate_demo_data.py
python scripts/run_preprocessing.py
python scripts/run_registration.py
python scripts/run_validation.py
```

Expected demo artifacts:
- `data/preprocessed/sample_fixed.nii.gz`
- `data/preprocessed/sample_moving.nii.gz`
- `data/masks/sample_calcium_mask.nii.gz`
- `outputs/centerlines_warped/sample_centerline_mask.nii.gz`
- `docs/images/within_10mm_by_seed.png`
- `docs/images/mean_distance_by_seed.png`
- `docs/images/runtime_by_seed.png`
- `experiments/registration_comparison.csv`
- `experiments/registration_comparison_summary.csv`

## Example Outputs

### Percent Within 10mm by Seed (42–66)

![Percent within 10mm by seed](docs/images/within_10mm_by_seed.png)

### Mean Distance by Seed (42–66)

![Mean distance by seed](docs/images/mean_distance_by_seed.png)

### Runtime by Seed (42–66)

![Runtime by seed](docs/images/runtime_by_seed.png)

## Latest Registration Results (Seeds 42–66)

Metrics below are populated from `experiments/registration_comparison_summary.csv`.

| Metric | Value |
|---|---:|
| Mean distance (mean across seeds) | 6.8625 mm |
| Mean distance (median across seeds) | 6.5412 mm |
| Percent within 10mm (mean) | 74.73% |
| Percent within 10mm (median) | 74.31% |
| Runtime per scan (mean) | 2.1988 sec |

## Registration Accuracy and Stability

We use a rigid initialization followed by affine refinement, with a lightweight retry mechanism for failed cases.

Observed synthetic evaluation metrics remain stable across seed sweeps, with runtime and proximity metrics written to `experiments/registration_comparison.csv` and `experiments/registration_comparison_summary.csv`.

| Registration Strategy | Runtime/scan (mean) | Mean Distance (mm, mean) | %Within10mm (mean) |
|---|---:|---:|---:|
| Rigid + Affine (`rigid_affine`) | 2.1988 | 6.8625 | 74.73 |

The pipeline reports fresh per-seed metrics and an aggregate summary for this single strategy to track both alignment quality and runtime consistency.

## Config-Driven Commands

Preprocessing config: `configs/preprocessing.yaml`

Registration configs:
- `configs/registration/affine.yaml`

Validation and segmentation config: `configs/segmentation.yaml`

## Testing

```bash
pytest -q
```

## Notes on Reuse of Existing Code

This implementation extends the existing repository and retains legacy scripts under `coca_project/src/`. The new `src/predict_cac/` package provides modular interfaces to standardize experiments while preserving prior work.

If real COCA data is present, ingestion and preprocessing continue to process those files
normally. The synthetic generator only provides a fallback dataset for demonstrations.
