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
    grouped = results_df.groupby("strategy", as_index=False).agg(
        mean_distance_mm_mean=("mean_distance_mm", "mean"),
        mean_distance_mm_median=("mean_distance_mm", "median"),
        mean_distance_mm_std=("mean_distance_mm", "std"),
        percent_within_10mm_mean=("percent_within_10mm", "mean"),
        percent_within_10mm_median=("percent_within_10mm", "median"),
        percent_within_10mm_std=("percent_within_10mm", "std"),
        runtime_sec_mean=("runtime_sec", "mean"),
        runtime_sec_median=("runtime_sec", "median"),
        runtime_sec_std=("runtime_sec", "std"),
    )
    return grouped.sort_values("strategy").reset_index(drop=True)


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


def _extract_points_physical(mask_path: Path) -> tuple[np.ndarray, sitk.Image]:
    mask_image = sitk.ReadImage(str(mask_path))
    array_zyx = sitk.GetArrayFromImage(mask_image)
    indices_zyx = np.argwhere(array_zyx > 0)
    points: list[tuple[float, float, float]] = []
    for z, y, x in indices_zyx:
        points.append(mask_image.TransformIndexToPhysicalPoint((int(x), int(y), int(z))))
    return np.asarray(points, dtype=np.float64), mask_image


def _print_image_geometry(label: str, image: sitk.Image) -> None:
    print(
        f"[GEOM] {label} spacing={tuple(float(v) for v in image.GetSpacing())} "
        f"origin={tuple(float(v) for v in image.GetOrigin())} "
        f"direction={tuple(float(v) for v in image.GetDirection())} "
        f"size={tuple(int(v) for v in image.GetSize())}"
    )


def _count_points_in_bounds(points_physical: np.ndarray, reference_image: sitk.Image) -> int:
    if len(points_physical) == 0:
        return 0
    size = reference_image.GetSize()
    valid = 0
    for point in points_physical:
        try:
            x, y, z = reference_image.TransformPhysicalPointToIndex(tuple(float(v) for v in point))
        except RuntimeError:
            continue
        if 0 <= x < size[0] and 0 <= y < size[1] and 0 <= z < size[2]:
            valid += 1
    return valid


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

        fixed_pre_img = sitk.ReadImage(str(fixed_preprocessed))
        moving_pre_img = sitk.ReadImage(str(moving_preprocessed))
        fixed_center_img = sitk.ReadImage(str(fixed_centerline_mask_path))
        moving_center_img = sitk.ReadImage(str(moving_centerline_mask_path))
        print(f"\n[DEBUG] Seed={seed} coordinate system inspection")
        _print_image_geometry("fixed_preprocessed", fixed_pre_img)
        _print_image_geometry("moving_preprocessed", moving_pre_img)
        _print_image_geometry("fixed_centerline_mask", fixed_center_img)
        _print_image_geometry("moving_centerline_mask", moving_center_img)

        moving_centerline_points, _ = _extract_points_physical(moving_centerline_mask_path)
        fixed_centerline_points, fixed_centerline_image = _extract_points_physical(fixed_centerline_mask_path)

        print(f"\n[SEED={seed}] Starting registration pipeline...")
        print(f"[SEED={seed}] Moving centerline points: {len(moving_centerline_points)}")
        print(f"[SEED={seed}] Fixed centerline points: {len(fixed_centerline_points)}")

        # ========== TRY RIGID+AFFINE FIRST ==========
        print(f"\n[SEED={seed}] === PHASE 1: Rigid+Affine Registration ===")
        strategy_reg_dir = output_reg_dir / f"seed_{seed}" / "rigid_affine"
        start = time.perf_counter()
        reg_result = run_registration(
            moving_image_path=moving_preprocessed,
            fixed_image_path=fixed_preprocessed,
            out_dir=strategy_reg_dir,
            strategy="rigid_affine",
            levels=int(reg_cfg.get("levels", 3)),
            iterations=[int(v) for v in reg_cfg.get("iterations", [1000, 500, 200])],
            shrink_factors=[int(v) for v in reg_cfg.get("shrink_factors", [4, 2, 1])],
            smoothing_sigmas=[float(v) for v in reg_cfg.get("smoothing_sigmas", [2, 1, 0])],
            seed=seed,
        )
        rigid_affine_runtime = float(time.perf_counter() - start)

        print(f"[SEED={seed}] rigid_transform: {_transform_signature(reg_result.get('rigid_transform'))}")
        print(f"[SEED={seed}] affine_transform: {_transform_signature(reg_result.get('affine_transform'))}")

        affine_path = reg_result.get("affine_transform")
        if affine_path is None:
            raise RuntimeError(f"Affine transform missing for seed {seed}.")

        composite_tx = _compose_rigid_affine_transform(
            rigid_transform_path=str(reg_result["rigid_transform"]),
            affine_transform_path=str(affine_path),
        )

        transformed_points = np.asarray(
            [composite_tx.TransformPoint(tuple(float(v) for v in point)) for point in moving_centerline_points],
            dtype=np.float64,
        )
        in_bounds = _count_points_in_bounds(transformed_points, fixed_centerline_image)
        print(f"[SEED={seed}] Transformed points in bounds: {in_bounds}/{len(transformed_points)}")

        transformed_mask_path = transformed_centerline_root / f"seed_{seed}" / "rigid_affine_centerline_mask.nii.gz"
        transformed_mask_path, transformed_count = _resample_centerline_mask(
            moving_centerline_mask_path=moving_centerline_mask_path,
            fixed_reference_image=fixed_centerline_image,
            moving_to_fixed_transform=composite_tx,
            out_path=transformed_mask_path,
        )
        print(f"[SEED={seed}] Resampled centerline points in mask: {transformed_count}")

        # Initialize default values
        affine_mean_dist = 999.0
        affine_within_10mm = 0.0
        use_rigid_fallback = False

        if transformed_count == 0:
            print(f"[SEED={seed}] WARNING: Empty transformed centerline mask with rigid_affine!")
            print(f"[SEED={seed}] Will fall back to rigid-only registration.")
            use_rigid_fallback = True
        else:
            transformed_mask_points, _ = _extract_points_physical(transformed_mask_path)
            sample_count = min(5, len(transformed_mask_points))
            print(f"[SEED={seed}] Transformed centerline points: {len(transformed_mask_points)}")
            print(f"[SEED={seed}] Sample transformed points: {transformed_mask_points[:sample_count].tolist()}")

            eval_result = evaluate_registration(
                calcium_mask_path=calcium_mask_path,
                centerline_mask_path=transformed_mask_path,
                threshold_mm=float(args.threshold_mm),
            )
            affine_mean_dist = float(eval_result["mean_distance_mm"])
            affine_within_10mm = float(eval_result["percent_within_10mm"])
            print(
                f"[SEED={seed}] rigid_affine metric: "
                f"mean_distance_mm={affine_mean_dist:.6f} "
                f"percent_within_10mm={affine_within_10mm:.6f}"
            )

            # Check for poor registration indicators
            if affine_within_10mm < 30.0:
                print(f"[SEED={seed}] ALERT: Low within_10mm ({affine_within_10mm:.2f}%) - potential affine failure")
                use_rigid_fallback = True
            elif affine_mean_dist > 15.0:
                print(f"[SEED={seed}] ALERT: High mean distance ({affine_mean_dist:.2f}mm) - potential affine failure")
                use_rigid_fallback = True

        # ========== IF AFFINE SUSPICIOUS, TRY RIGID-ONLY ==========
        if use_rigid_fallback:
            print(f"\n[SEED={seed}] === PHASE 2: Fallback to Rigid-Only Registration ===")
            rigid_reg_dir = output_reg_dir / f"seed_{seed}" / "rigid_only"
            start = time.perf_counter()
            rigid_result = run_registration(
                moving_image_path=moving_preprocessed,
                fixed_image_path=fixed_preprocessed,
                out_dir=rigid_reg_dir,
                strategy="rigid",
                levels=int(reg_cfg.get("levels", 3)),
                iterations=[int(v) for v in reg_cfg.get("iterations", [1000, 500, 200])],
                shrink_factors=[int(v) for v in reg_cfg.get("shrink_factors", [4, 2, 1])],
                smoothing_sigmas=[float(v) for v in reg_cfg.get("smoothing_sigmas", [2, 1, 0])],
                seed=seed,
            )
            rigid_runtime = float(time.perf_counter() - start)

            rigid_tx = sitk.ReadTransform(str(rigid_result["rigid_transform"]))
            rigid_transformed_points = np.asarray(
                [rigid_tx.TransformPoint(tuple(float(v) for v in point)) for point in moving_centerline_points],
                dtype=np.float64,
            )
            rigid_in_bounds = _count_points_in_bounds(rigid_transformed_points, fixed_centerline_image)
            print(f"[SEED={seed}] Rigid: transformed points in bounds: {rigid_in_bounds}/{len(rigid_transformed_points)}")

            rigid_transformed_mask_path = transformed_centerline_root / f"seed_{seed}" / "rigid_only_centerline_mask.nii.gz"
            rigid_transformed_mask_path, rigid_transformed_count = _resample_centerline_mask(
                moving_centerline_mask_path=moving_centerline_mask_path,
                fixed_reference_image=fixed_centerline_image,
                moving_to_fixed_transform=rigid_tx,
                out_path=rigid_transformed_mask_path,
            )
            print(f"[SEED={seed}] Rigid: resampled centerline points: {rigid_transformed_count}")

            if rigid_transformed_count > 0:
                rigid_eval_result = evaluate_registration(
                    calcium_mask_path=calcium_mask_path,
                    centerline_mask_path=rigid_transformed_mask_path,
                    threshold_mm=float(args.threshold_mm),
                )
                rigid_mean_dist = float(rigid_eval_result["mean_distance_mm"])
                rigid_within_10mm = float(rigid_eval_result["percent_within_10mm"])
                print(
                    f"[SEED={seed}] rigid metric: "
                    f"mean_distance_mm={rigid_mean_dist:.6f} "
                    f"percent_within_10mm={rigid_within_10mm:.6f}"
                )

                # Compare and choose best strategy
                if use_rigid_fallback and rigid_within_10mm > affine_within_10mm:
                    print(f"\n[SEED={seed}] DECISION: Using rigid-only (better than affine)")
                    final_result = {
                        "strategy": "rigid",
                        "seed": int(seed),
                        "mean_distance_mm": rigid_mean_dist,
                        "percent_within_10mm": rigid_within_10mm,
                        "runtime_sec": rigid_runtime,
                    }
                    fallback_used = True
                else:
                    print(f"\n[SEED={seed}] DECISION: Using rigid_affine despite concerns")
                    final_result = {
                        "strategy": "rigid_affine",
                        "seed": int(seed),
                        "mean_distance_mm": affine_mean_dist,
                        "percent_within_10mm": affine_within_10mm,
                        "runtime_sec": rigid_affine_runtime,
                    }
                    fallback_used = False
            else:
                print(f"[SEED={seed}] Rigid also failed (empty mask). Using affine with warning.")
                final_result = {
                    "strategy": "rigid_affine",
                    "seed": int(seed),
                    "mean_distance_mm": affine_mean_dist,
                    "percent_within_10mm": affine_within_10mm,
                    "runtime_sec": rigid_affine_runtime,
                }
                fallback_used = False
        else:
            print(f"\n[SEED={seed}] DECISION: rigid_affine metrics acceptable - using result")
            final_result = {
                "strategy": "rigid_affine",
                "seed": int(seed),
                "mean_distance_mm": affine_mean_dist,
                "percent_within_10mm": affine_within_10mm,
                "runtime_sec": rigid_affine_runtime,
            }
            fallback_used = False

        print(f"[SEED={seed}] Final metrics: {final_result}")
        rows.append(final_result)

    results_df = pd.DataFrame(rows, columns=["strategy", "seed", "mean_distance_mm", "percent_within_10mm", "runtime_sec"])
    results_df = results_df.sort_values(["strategy", "seed"]).reset_index(drop=True)

    ensure_dir(args.output_csv.parent)
    results_df.to_csv(args.output_csv, index=False)

    summary_df = _summary_dataframe(results_df)
    ensure_dir(args.summary_csv.parent)
    summary_df.to_csv(args.summary_csv, index=False)

    print("\n=== Registration Strategy Comparison Summary ===")
    print(_format_console_table(summary_df))

    print(f"\nSaved per-run results to: {args.output_csv}")
    print(f"Saved per-strategy summary to: {args.summary_csv}")


if __name__ == "__main__":
    main()
