"""Atlas registration pipeline using SimpleITK rigid + affine stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import SimpleITK as sitk

from predict_cac.registration.registration_utils import read_image, save_transform
from predict_cac.utils.io import ensure_dir


VALID_STRATEGIES = {"rigid_affine"}


def normalize_strategy(strategy: str) -> str:
    value = str(strategy).strip().lower()
    aliases = {
        "affine": "rigid_affine",
    }
    normalized = aliases.get(value, value)
    if normalized not in VALID_STRATEGIES:
        raise ValueError(f"Unsupported registration strategy: {strategy}")
    return normalized


def _registration_method(
    metric_sampling_percentage: float = 1.0,
    iterations: Sequence[int] = (1000, 500, 200),
    shrink_factors: Sequence[int] = (4, 2, 1),
    smoothing_sigmas: Sequence[float] = (2.0, 1.0, 0.0),
    seed: int = 42,
) -> sitk.ImageRegistrationMethod:
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMeanSquares()
    method.SetMetricSamplingStrategy(method.NONE)
    method.SetInterpolator(sitk.sitkLinear)

    method.SetShrinkFactorsPerLevel([int(v) for v in shrink_factors])
    method.SetSmoothingSigmasPerLevel([float(v) for v in smoothing_sigmas])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    total_iterations = int(sum(int(v) for v in iterations))
    method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=total_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    return method


def _execute_with_retry(
    method: sitk.ImageRegistrationMethod,
    fixed: sitk.Image,
    moving: sitk.Image,
    fallback_method: sitk.ImageRegistrationMethod | None = None,
) -> sitk.Transform:
    try:
        return method.Execute(fixed, moving)
    except RuntimeError:
        if fallback_method is None:
            raise
        return fallback_method.Execute(fixed, moving)


def run_registration(
    moving_image_path: Path,
    fixed_image_path: Path,
    out_dir: Path,
    strategy: str = "rigid_affine",
    levels: int = 3,
    iterations: Sequence[int] = (1000, 500, 200),
    shrink_factors: Sequence[int] = (4, 2, 1),
    smoothing_sigmas: Sequence[float] = (2.0, 1.0, 0.0),
    seed: int = 42,
) -> dict[str, Any]:
    """Run rigid initialization followed by affine refinement."""
    ensure_dir(out_dir)
    fixed = read_image(fixed_image_path)
    moving = read_image(moving_image_path)

    normalize_strategy(strategy)
    warnings: list[str] = []

    rigid_init = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    pyramid_shrink = tuple(shrink_factors[:levels])
    pyramid_smooth = tuple(smoothing_sigmas[:levels])
    pyramid_iters = tuple(iterations[:levels])

    rigid_method = _registration_method(
        metric_sampling_percentage=1.0,
        iterations=pyramid_iters,
        shrink_factors=pyramid_shrink,
        smoothing_sigmas=pyramid_smooth,
        seed=seed,
    )
    rigid_method.SetInitialTransform(rigid_init, inPlace=False)
    rigid_fallback_method = _registration_method(
        metric_sampling_percentage=1.0,
        iterations=pyramid_iters,
        shrink_factors=pyramid_shrink,
        smoothing_sigmas=pyramid_smooth,
        seed=seed,
    )
    rigid_fallback_method.SetInitialTransform(rigid_init, inPlace=False)
    try:
        rigid_tx = _execute_with_retry(rigid_method, fixed, moving, fallback_method=rigid_fallback_method)
    except RuntimeError:
        rigid_tx = sitk.Transform(rigid_init)
        warnings.append("rigid_registration_failed_using_initializer")
    rigid_path = save_transform(rigid_tx, out_dir / "rigid.tfm")

    affine_init = sitk.AffineTransform(3)
    affine_method = _registration_method(
        metric_sampling_percentage=1.0,
        iterations=pyramid_iters,
        shrink_factors=pyramid_shrink,
        smoothing_sigmas=pyramid_smooth,
        seed=seed,
    )
    affine_method.SetMovingInitialTransform(rigid_tx)
    affine_method.SetInitialTransform(affine_init, inPlace=False)
    affine_fallback_method = _registration_method(
        metric_sampling_percentage=1.0,
        iterations=pyramid_iters,
        shrink_factors=pyramid_shrink,
        smoothing_sigmas=pyramid_smooth,
        seed=seed,
    )
    affine_fallback_method.SetMovingInitialTransform(rigid_tx)
    affine_fallback_method.SetInitialTransform(affine_init, inPlace=False)
    try:
        affine_tx = _execute_with_retry(
            affine_method,
            fixed,
            moving,
            fallback_method=affine_fallback_method,
        )
    except RuntimeError:
        affine_tx = sitk.AffineTransform(3)
        warnings.append("affine_registration_failed_using_identity")
    affine_path = save_transform(affine_tx, out_dir / "affine.tfm")


    return {
        "strategy": "rigid_affine",
        "rigid_transform": str(rigid_path),
        "affine_transform": str(affine_path),
        "rigid_parameter_count": int(len(rigid_tx.GetParameters())),
        "affine_parameter_count": int(len(affine_tx.GetParameters())),
        "warnings": warnings,
        "metric": "MeanSquares",
        "levels": levels,
        "iterations": [int(v) for v in pyramid_iters],
        "shrink_factors": [int(v) for v in pyramid_shrink],
        "smoothing_sigmas": [float(v) for v in pyramid_smooth],
        "seed": int(seed),
    }
