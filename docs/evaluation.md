# Evaluation Summary (Rigid+Affine Only)

This project uses a single registration strategy: **rigid initialization followed by affine refinement**, with a lightweight retry mechanism for failed cases.

## Latest Multi-Seed Results

Latest validated run covers seeds **42–66** (25 scans), with metrics from `experiments/registration_comparison_summary.csv`.

| Metric | Value |
|---|---:|
| Strategy | `rigid_affine` |
| Mean distance (mean) | 6.8625 mm |
| Mean distance (median) | 6.5412 mm |
| Mean distance (std) | 2.5299 mm |
| Percent within 10mm (mean) | 74.73% |
| Percent within 10mm (median) | 74.31% |
| Percent within 10mm (std) | 13.55% |
| Runtime per scan (mean) | 2.1988 sec |
| Runtime per scan (median) | 1.9897 sec |

## Plots from Latest Run

### Percent Within 10mm by Seed

![Percent within 10mm by seed](images/within_10mm_by_seed.png)

### Mean Distance by Seed

![Mean distance by seed](images/mean_distance_by_seed.png)

### Runtime by Seed

![Runtime by seed](images/runtime_by_seed.png)

## Output Files

- `experiments/registration_comparison.csv`
- `experiments/registration_comparison_summary.csv`
- `docs/images/within_10mm_by_seed.png`
- `docs/images/mean_distance_by_seed.png`
- `docs/images/runtime_by_seed.png`
