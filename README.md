# GSoC 2026 — ML4SCI PrediCT Project 1: Heart Segmentation Pipeline

## Overview
This repository implements a fast, reproducible heart segmentation pipeline on the Stanford COCA cardiac CT dataset. A lightweight 3D U-Net trained on TotalSegmentator-generated ground truth achieves validation Dice of 0.919 and test Dice of 0.833 on 17 held-out scans, running ~296x faster than TotalSegmentator at inference.

## Results Summary
| Metric | Value |
| --- | --- |
| Total scans | 64 |
| Train / Val / Test | 38 / 9 / 17 |
| Best Validation Dice | 0.9192 |
| Test Dice mean±std | 0.833 ± 0.030 |
| Test Dice min / max | 0.781 / 0.894 |
| Model inference time | ~0.21s per scan |
| TotalSegmentator time | ~60s per scan |
| Speedup | ~296x faster |
| Model parameters | ~3.5M |

## Repository Structure
```
predict-cac-gsoc/
├── config.yaml                         # Training, preprocessing, benchmark settings
├── requirements.txt                    # Frozen runtime dependencies
├── scripts/
│   ├── benchmark_inference.py          # Model vs TotalSegmentator benchmark
│   ├── generate_heart_labels.py        # Produce TS heart masks
│   ├── verify_labels.py                # Label QC helpers
│   ├── train_heart_model.py            # Training entrypoint
│   └── download_coca.py                # (Optional) dataset fetch helper
├── src/predict_cac/
│   ├── utils/config_loader.py          # YAML loader
│   └── segmentation/
│       ├── datasets/heart_dataset.py   # Dataset/dataloader
│       ├── models/unet_3d.py           # Lightweight 3D U-Net
│       ├── training/                   # Losses and train loop
│       ├── evaluation/metrics.py       # Metric utilities
│       ├── inference/predict.py        # Single/batch inference
│       └── utils/
│           ├── preprocessing.py        # MONAI transforms
│           └── visualization.py        # Plots and overlays
├── notebooks/heart_segmentation_eval.ipynb  # Evaluation summary notebook
├── data/coca_ct/                       # Raw CTs (not tracked)
└── outputs/                            # Generated artifacts (not tracked)
	├── labels/                         # TS-derived heart masks
	├── label_qc/                       # QC snapshots
	├── checkpoints/                    # Model weights
	├── metrics/                        # CSV/JSON metrics, benchmarks
	└── figures/                        # Plots and overlays
```

## Reproduction Steps
1. `python scripts/generate_heart_labels.py --input_dir data/coca_ct --output_dir outputs/labels --license_number YOUR_KEY`
2. `python scripts/verify_labels.py --label_dir outputs/labels --output_dir outputs/label_qc`
3. `python scripts/train_heart_model.py --config config.yaml`
4. `python scripts/benchmark_inference.py --config config.yaml --checkpoint outputs/checkpoints/best_model.pth --license_number YOUR_KEY`
5. `jupyter notebook notebooks/heart_segmentation_eval.ipynb`

## Environment Setup
1. `python -m venv .venv`
2. `.venv\Scripts\activate`
3. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
4. `pip install monai[all] nibabel SimpleITK numpy pandas matplotlib seaborn scikit-learn scipy tqdm pyyaml TotalSegmentator jupyter ipykernel`
5. `python -m ipykernel install --user --name=predict_cac --display-name "PrediCT CAC (venv)"`

## Label Generation Details
- TotalSegmentator v2, task: heartchambers_highres
- License: academic/research (non-commercial)
- Labels merged: heart_myocardium, atrium_left, atrium_right, ventricle_left, ventricle_right, aorta, pulmonary_artery
- Labels NOT redistributed — only binary merged masks saved

## Justification
"We built a 3D segmentation preprocessing pipeline using MONAI on 64 cardiac CT scans from the Stanford COCA dataset. The pipeline applies HU windowing [-200,600] to isolate soft tissue, resamples all volumes to 2.5mm isotropic spacing to normalize variable axial resolution across scanners, and standardizes orientation to RAS. A patient-level 60/15/25 train/val/test split with fixed seed ensures no data leakage. Class imbalance (heart voxels ~2.8% of total) is addressed through foreground-weighted DiceCE loss rather than oversampling, which preserves spatial context critical for 3D segmentation.

Augmentation includes random left-right flips, small rotations, Gaussian noise, zoom (±10%), and 3D elastic deformation. Elastic deformation is particularly important given the relatively small training set — it creates plausible new cardiac shapes and significantly reduces overfitting. The CacheDataset pre-loads all transforms into RAM, eliminating per-epoch I/O overhead. This pipeline is optimized for segmentation by operating on full 3D volumes rather than 2D slices, preserving inter-slice spatial relationships essential for accurate heart boundary delineation."

"We chose a lightweight 3D U-Net (3.5M parameters, channels 16-32-64-128) over heavier alternatives like nnU-Net because the task requires a fast preprocessing step rather than maximal segmentation accuracy. Operating at 2.5mm isotropic resolution the model runs ~296x faster than TotalSegmentator while maintaining validation Dice of 0.919 and test Dice of 0.833 across 17 unseen scans. The 3D architecture preserves inter-slice continuity critical for robust whole-heart localization given COCA variable axial spacing (1.5-3.0mm). The predicted heart mask serves as a downstream ROI for CAC segmentation, restricting analysis to the cardiac region and eliminating rib and spine false positives from HU>130 thresholding."

## Fair Comparison Protocol
- Same input CT and spacing (2.5mm isotropic), batch size 1, two warmup runs
- Same GPU (RTX 3050 6GB) and software stack; torch.cuda.synchronize for timing
- Model time includes preprocessing + inference + postprocessing
- Report per-scan table plus mean±std speedup (ts_time / model_time)

## Dataset Statistics (from outputs/metrics/dataset_stats.csv)
| Stat | Value |
| --- | --- |
| Total scans | 64 |
| Train / Val / Test | 38 / 9 / 17 |
| In-plane spacing mean / median | 0.383 mm / 0.373 mm |
| Through-plane spacing mean / median | 2.96 mm / 3.0 mm |
| Heart voxels (% of volume) mean ± std | 2.75% ± 0.81% |
| Heart voxels min / max | 0.78% / 6.25% |

## Deliverables
- Model weights: [best_model.pth (Google Drive)](https://drive.google.com/file/d/1avlC-tDXG1MKHCcjoF0ahYTM3MYrxmRN/view?usp=sharing)
- Per-case metrics: outputs/metrics/per_case_metrics.csv
- Benchmark results: outputs/metrics/benchmark_results.csv

## Acknowledgments
ML4SCI, Stanford COCA dataset, TotalSegmentator, MONAI
