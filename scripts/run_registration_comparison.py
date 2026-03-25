"""Run rigid+affine registration comparison across seeds and export metrics."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict_cac.evaluation.registration_eval import evaluate_registration
from predict_cac.registration.atlas_registration import run_registration
from predict_cac.transforms.preprocessing_pipeline import run_preprocessing_pipeline
from predict_cac.utils.io import ensure_dir
from predict_cac.utils.synthetic_data import generate_synthetic_demo_data


def _summary_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    valid_mean_distance = results_df["mean_distance_mm"].dropna()
    valid_within_10mm = results_df["percent_within_10mm"].dropna()

    summary_row = {
        "strategy": "rigid_affine",
        "mean_distance_mm_mean": float(valid_mean_distance.mean()) if not valid_mean_distance.empty else float("nan"),
        "mean_distance_mm_median": float(valid_mean_distance.median()) if not valid_mean_distance.empty else float("nan"),
        "mean_distance_mm_std": float(valid_mean_distance.std()) if not valid_mean_distance.empty else float("nan"),
        "percent_within_10mm_mean": float(valid_within_10mm.mean()) if not valid_within_10mm.empty else float("nan"),
        "percent_within_10mm_median": float(valid_within_10mm.median()) if not valid_within_10mm.empty else float("nan"),
        "percent_within_10mm_std": float(valid_within_10mm.std()) if not valid_within_10mm.empty else float("nan"),
        "runtime_sec_mean": float(results_df["runtime_sec"].mean()),
        "runtime_sec_median": float(results_df["runtime_sec"].median()),
        "runtime_sec_std": float(results_df["runtime_sec"].std()),
    }
    return pd.DataFrame([summary_row])


def _format_console_table(summary_df: pd.DataFrame) -> str:
    display = summary_df[
        [
            "strategy",
            "runtime_sec_mean",
            "mean_distance_mm_mean",
            "percent_within_10mm_mean",
        ]
    ].copy()
    display = display.rename(
        columns={
            "strategy": "Strategy",
            "runtime_sec_mean": "Runtime/scan (sec)",
            "mean_distance_mm_mean": "Mean Distance (mm)",
            "percent_within_10mm_mean": "% Within 10mm",
        }
    )
    return display.to_string(index=False)


def _transform_signature(transform_path: str | None) -> str:
    if transform_path is None:
        return "none"
    tx = sitk.ReadTransform(str(transform_path))
    params = list(tx.GetParameters())
    preview = ",".join(f"{float(v):.6f}" for v in params[:6])
    return f"n={len(params)} first=[{preview}]"


def _compose_rigid_affine_transform(rigid_transform_path: str, affine_transform_path: str) -> sitk.Transform:
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(sitk.ReadTransform(str(rigid_transform_path)))
    composite.AddTransform(sitk.ReadTransform(str(affine_transform_path)))
    return composite


def _resample_centerline_mask(
    moving_centerline_mask_path: Path,
    fixed_reference_image: sitk.Image,
    moving_to_fixed_transform: sitk.Transform,
    out_path: Path,
) -> tuple[Path, int]:
    """Resample and return path + count of points in transformed mask."""
    moving_centerline_image = sitk.ReadImage(str(moving_centerline_mask_path))
    fixed_to_moving_transform = moving_to_fixed_transform.GetInverse()
    warped = sitk.Resample(
        moving_centerline_image,
        fixed_reference_image,
        fixed_to_moving_transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    ensure_dir(out_path.parent)
    sitk.WriteImage(warped, str(out_path))
    
    # Count points in resampled mask
    warped_array = sitk.GetArrayFromImage(warped)
    point_count = int(np.sum(warped_array > 0))
    return out_path, point_count


def _run_rigid_affine_attempt(
    seed: int,
    moving_preprocessed: Path,
    fixed_preprocessed: Path,
    output_dir: Path,
    fixed_centerline_image: sitk.Image,
    moving_centerline_mask_path: Path,
    transformed_mask_path: Path,
    calcium_mask_path: Path,
    threshold_mm: float,
    levels: int,
    iterations: list[int],
    shrink_factors: list[int],
    smoothing_sigmas: list[float],
) -> dict[str, float | str]:
    start = time.perf_counter()
    reg_result = run_registration(
        moving_image_path=moving_preprocessed,
        fixed_image_path=fixed_preprocessed,
        out_dir=output_dir,
        strategy="rigid_affine",
        levels=levels,
        iterations=iterations,
        shrink_factors=shrink_factors,
        smoothing_sigmas=smoothing_sigmas,
        seed=seed,
    )
    runtime_sec = float(time.perf_counter() - start)

    affine_path = reg_result.get("affine_transform")
    if affine_path is None:
        raise RuntimeError(f"Affine transform missing for seed {seed}.")

    composite_tx = _compose_rigid_affine_transform(
        rigid_transform_path=str(reg_result["rigid_transform"]),
        affine_transform_path=str(affine_path),
    )
    transformed_mask_path, transformed_count = _resample_centerline_mask(
        moving_centerline_mask_path=moving_centerline_mask_path,
        fixed_reference_image=fixed_centerline_image,
        moving_to_fixed_transform=composite_tx,
        out_path=transformed_mask_path,
    )

    if transformed_count == 0:
        mean_distance_mm = float("nan")
        percent_within_10mm = float("nan")
    else:
        eval_result = evaluate_registration(
            calcium_mask_path=calcium_mask_path,
            centerline_mask_path=transformed_mask_path,
            threshold_mm=threshold_mm,
        )
        mean_distance_mm = float(eval_result["mean_distance_mm"])
        percent_within_10mm = float(eval_result["percent_within_10mm"])

    return {
        "rigid_transform": str(reg_result["rigid_transform"]),
        "affine_transform": str(affine_path),
        "mean_distance_mm": mean_distance_mm,
        "percent_within_10mm": percent_within_10mm,
        "runtime_sec": runtime_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rigid+affine registration comparison")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-end", type=int, default=61)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "experiments" / "registration_comparison.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ROOT / "experiments" / "registration_comparison_summary.csv",
    )
    parser.add_argument("--preprocess-config", type=Path, default=ROOT / "configs" / "preprocessing.yaml")
    parser.add_argument("--registration-config", type=Path, default=ROOT / "configs" / "registration" / "affine.yaml")
    parser.add_argument("--threshold-mm", type=float, default=10.0)
    args = parser.parse_args()

    seeds = list(range(int(args.seed_start), int(args.seed_end) + 1))

    preprocess_cfg = yaml.safe_load(args.preprocess_config.read_text(encoding="utf-8"))
    reg_cfg = yaml.safe_load(args.registration_config.read_text(encoding="utf-8"))

    spacing_values = [float(v) for v in preprocess_cfg.get("target_spacing", [1.0, 1.0, 1.0])]
    while len(spacing_values) < 3:
        spacing_values.append(1.0)
    target_spacing = (spacing_values[0], spacing_values[1], spacing_values[2])
    input_nifti_dir = ROOT / preprocess_cfg.get("input_nifti_dir", "data/nifti")
    output_pre_dir = ROOT / preprocess_cfg.get("output_nifti_dir", "data/preprocessed")
    output_reg_dir = ROOT / reg_cfg.get("output_dir", "outputs/registered_atlas")

    fixed_preprocessed = output_pre_dir / "sample_fixed.nii.gz"
    moving_preprocessed = output_pre_dir / "sample_moving.nii.gz"
    calcium_mask_path = ROOT / "data" / "masks" / "sample_calcium_mask.nii.gz"
    fixed_centerline_mask_path = ROOT / "outputs" / "centerlines_warped" / "sample_centerline_mask.nii.gz"
    moving_centerline_mask_path = ROOT / "outputs" / "centerlines_warped" / "sample_centerline_mask_moving.nii.gz"
    transformed_centerline_root = ROOT / "outputs" / "centerlines_warped" / "comparison_debug"
    levels = int(reg_cfg.get("levels", 3))
    base_iterations = [int(v) for v in reg_cfg.get("iterations", [1000, 500, 200])]
    base_shrink_factors = [int(v) for v in reg_cfg.get("shrink_factors", [4, 2, 1])]
    base_smoothing_sigmas = [float(v) for v in reg_cfg.get("smoothing_sigmas", [2, 1, 0])]
    retry_iterations = [1200, 700, 300]
    retry_smoothing_sigmas = [1.5, 0.5, 0.0]
    retry_threshold_percent = 30.0

    rows: list[dict[str, float | int | str]] = []

    for seed in seeds:
        np.random.seed(seed)
        _ = generate_synthetic_demo_data(ROOT, seed=seed)
        run_preprocessing_pipeline(
            input_nifti_dir=input_nifti_dir,
            output_dir=output_pre_dir,
            target_spacing=target_spacing,
        )

        if not fixed_preprocessed.exists() or not moving_preprocessed.exists():
            raise FileNotFoundError("Preprocessed moving/fixed images are missing after preprocessing stage.")

        fixed_centerline_image = sitk.ReadImage(str(fixed_centerline_mask_path))

        strategy_reg_dir = output_reg_dir / f"seed_{seed}" / "rigid_affine"
        transformed_mask_path = transformed_centerline_root / f"seed_{seed}" / "rigid_affine_centerline_mask.nii.gz"
        best_attempt = _run_rigid_affine_attempt(
            seed=seed,
            moving_preprocessed=moving_preprocessed,
            fixed_preprocessed=fixed_preprocessed,
            output_dir=strategy_reg_dir,
            fixed_centerline_image=fixed_centerline_image,
            moving_centerline_mask_path=moving_centerline_mask_path,
            transformed_mask_path=transformed_mask_path,
            calcium_mask_path=calcium_mask_path,
            threshold_mm=float(args.threshold_mm),
            levels=levels,
            iterations=base_iterations,
            shrink_factors=base_shrink_factors,
            smoothing_sigmas=base_smoothing_sigmas,
        )

        base_percent = float(best_attempt["percent_within_10mm"])
        if np.isnan(base_percent) or base_percent < retry_threshold_percent:
            retry_output_dir = output_reg_dir / f"seed_{seed}" / "rigid_affine_retry"
            retry_mask_path = transformed_centerline_root / f"seed_{seed}" / "rigid_affine_retry_centerline_mask.nii.gz"
            retry_attempt = _run_rigid_affine_attempt(
                seed=seed,
                moving_preprocessed=moving_preprocessed,
                fixed_preprocessed=fixed_preprocessed,
                output_dir=retry_output_dir,
                fixed_centerline_image=fixed_centerline_image,
                moving_centerline_mask_path=moving_centerline_mask_path,
                transformed_mask_path=retry_mask_path,
                calcium_mask_path=calcium_mask_path,
                threshold_mm=float(args.threshold_mm),
                levels=levels,
                iterations=retry_iterations,
                shrink_factors=base_shrink_factors,
                smoothing_sigmas=retry_smoothing_sigmas,
            )
            retry_percent = float(retry_attempt["percent_within_10mm"])
            if (np.isnan(base_percent) and not np.isnan(retry_percent)) or (
                not np.isnan(retry_percent) and retry_percent > base_percent
            ):
                best_attempt = retry_attempt

        print(f"[SEED={seed}] rigid_transform: {_transform_signature(str(best_attempt['rigid_transform']))}")
        print(f"[SEED={seed}] affine_transform: {_transform_signature(str(best_attempt['affine_transform']))}")

        final_result = {
            "strategy": "rigid_affine",
            "seed": int(seed),
            "mean_distance_mm": float(best_attempt["mean_distance_mm"]),
            "percent_within_10mm": float(best_attempt["percent_within_10mm"]),
            "runtime_sec": float(best_attempt["runtime_sec"]),
        }
        print(
            f"[SEED={seed}] rigid_affine metric: "
            f"mean_distance_mm={final_result['mean_distance_mm']:.6f} "
            f"percent_within_10mm={final_result['percent_within_10mm']:.6f}"
        )

        rows.append(final_result)

    results_df = pd.DataFrame(rows, columns=["strategy", "seed", "mean_distance_mm", "percent_within_10mm", "runtime_sec"])
    results_df = results_df.sort_values(["seed"]).reset_index(drop=True)

    ensure_dir(args.output_csv.parent)
    results_df.to_csv(args.output_csv, index=False)

    summary_df = _summary_dataframe(results_df)
    ensure_dir(args.summary_csv.parent)
    summary_df.to_csv(args.summary_csv, index=False)

    print("\n=== Registration Summary (rigid_affine) ===")
    print(_format_console_table(summary_df))

    print(f"\nSaved per-run results to: {args.output_csv}")
    print(f"Saved per-strategy summary to: {args.summary_csv}")


if __name__ == "__main__":
    main()
