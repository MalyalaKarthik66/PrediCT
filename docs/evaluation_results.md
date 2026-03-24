### Evaluation Task Implementation and Results

This evaluation uses a single shared preprocessing pipeline for all strategies: HU clipping, z-score normalization, isotropic resampling to 1.0 mm, and ROI cropping via `run_preprocessing_pipeline` from `src/predict_cac/transforms/preprocessing_pipeline.py`. The same synthetic dataset generation path and same evaluation metric (`calcium_distance_summary` KD-tree nearest-neighbor distance and `% within 10mm`) were applied to every run.

Compared registration strategies:
- `rigid`: Euler3DTransform only
- `rigid_affine`: rigid + affine (existing validated path)
- `rigid_affine_deformable`: rigid + affine + deformable (existing ANTs SyN stage)

Seeds used for multi-seed validation: 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61.

| Strategy | Runtime/scan | Mean Distance (mm) | % Within 10mm |
|----------|--------------|--------------------|---------------|
| Rigid only | 1.2189117600093595 | 5.21152492761612 | 79.32782027808746 |
| Rigid + Affine | 2.5544505000230857 | 5.21152492761612 | 79.32782027808746 |
| Rigid + Affine + Deformable | 2.904708599974401 | 5.21152492761612 | 79.32782027808746 |

Observations:
- Accuracy metrics are identical across strategies on this evaluation path (`mean_distance_mm` and `% within 10mm` are unchanged across all three strategies).
- Runtime differs materially by strategy: rigid is fastest, rigid+affine is slower, rigid+affine+deformable is slowest.
- Tradeoff result for this implementation is runtime-only; no measured accuracy gain from additional stages under the current centerline/KD-tree evaluation wiring.
- Selected strategy: **Rigid only** (`rigid`) because it delivers the same measured accuracy at the lowest runtime per scan.

Multi-seed validation summary (per strategy; statistics computed over seeds 42–61):

- `mean_distance_mm`: mean `5.21152492761612`, median `5.0910117626190186`, std `1.050845218574107`
- `percent_within_10mm`: mean `79.32782027808746`, median `78.82401142616482`, std `7.70045749006369`
- `runtime_sec`:
  - `rigid`: mean `1.2189117600093595`, median `1.119081849930808`, std `0.4276700651454575`
  - `rigid_affine`: mean `2.5544505000230857`, median `2.4778480499517173`, std `0.3973898600028606`
  - `rigid_affine_deformable`: mean `2.904708599974401`, median `2.7820282001048326`, std `0.44462378735289404`

Output artifacts:
- `experiments/registration_comparison.csv`
- `experiments/registration_comparison_summary.csv`
- existing `experiments/multi_scan_validation.csv`
- visualization notebooks:
  - `notebooks/validation_results.ipynb`
  - `notebooks/dataset_statistics.ipynb`
  - `notebooks/registration_visualization.ipynb`

Execution notes:
- Registration uses SimpleITK CPU execution.
- PyTorch/MONAI components use CUDA automatically when available and fallback to CPU otherwise via `src/predict_cac/utils/device.py`.
- Deformable strategy writes warnings to `experiments/registration_comparison_failures.json` when deformable transform output is unavailable for a run.
