"""Versioned FMT feature recipes for variable-scale Task5 primitives.

The legacy Task3 kinematic block differentiates with respect to sampled point
index.  That is harmless when every primitive has one fixed temporal scale,
but Task5 mixes integration step sizes and step counts in one batch.  Recipes
whose name contains ``physical_kinematic`` reconstruct each primitive's exact
sample times and therefore differentiate in physical time.
"""

from __future__ import annotations

import numpy as np
import torch

from FMT_Utils.DFT_FMT_3D import (
    fmt_feature_indices_3d,
    pathline_velocity_gradient_dft_features_3d,
    time_local_gram_dft_features_3d,
)


_EXACT_RECIPES = {
    "time_local_gram": (None, True, None),
    "kinematic": (None, False, "index"),
    "physical_kinematic": (None, False, "physical"),
    "gram_kinematic": (None, True, "index"),
    "gram_physical_kinematic": (None, True, "physical"),
}


def parse_task5_feature_recipe(name: str) -> tuple[str | None, bool, str | None]:
    """Return ``(cached_subset, include_gram, kinematic_time_basis)``."""
    name = str(name)
    if name in _EXACT_RECIPES:
        return _EXACT_RECIPES[name]
    suffixes = (
        ("_plus_gram_physical_kinematic", True, "physical"),
        ("_plus_gram_kinematic", True, "index"),
        ("_plus_time_local_gram", True, None),
        ("_plus_physical_kinematic", False, "physical"),
        ("_plus_kinematic", False, "index"),
    )
    for suffix, include_gram, time_basis in suffixes:
        if name.endswith(suffix):
            base = name[:-len(suffix)]
            if not base:
                raise ValueError(f"missing cached FMT subset in recipe {name!r}")
            # Validate the semantic subset immediately.
            fmt_feature_indices_3d(base)
            return base, include_gram, time_basis
    fmt_feature_indices_3d(name)
    return name, False, None


def task5_sample_times(integration_steps, physical_dt, sampled_steps: int) -> np.ndarray:
    """Reconstruct the exact rounded integration samples used by the cache."""
    steps = np.asarray(integration_steps, dtype=np.int64)
    dt = np.asarray(physical_dt, dtype=np.float64)
    if steps.ndim != 1 or dt.shape != steps.shape:
        raise ValueError("integration_steps and physical_dt must be matching vectors")
    if np.any(steps < int(sampled_steps) - 1) or np.any(dt <= 0):
        raise ValueError("invalid integration scale metadata")
    fractions = np.linspace(0.0, 1.0, int(sampled_steps), dtype=np.float64)
    sample_indices = np.rint(steps[:, None] * fractions[None, :])
    times = sample_indices * dt[:, None]
    if np.any(np.diff(times, axis=1) <= 0):
        raise ValueError("rounded Task5 sample times are not strictly increasing")
    return times


def task5_fmt_features_from_cache(
    source,
    sampled_steps: int,
    recipe: str,
    *,
    gram_num_freq: int = 2,
    kinematic_num_freq: int = 6,
    gram_subtract_initial: bool = True,
    gram_normalize_initial_scale: bool = True,
    kinematic_log_compress: bool = False,
    kinematic_pinv_rtol: float = 1e-6,
) -> np.ndarray:
    """Build one Task5 FMT recipe from an open ``numpy.load`` cache."""
    raw = np.asarray(source["raw_features"], dtype=np.float32)
    expected_width = 7 * int(sampled_steps) * 3
    if raw.ndim != 2 or raw.shape[1] != expected_width:
        raise ValueError(f"expected raw feature width {expected_width}, got {raw.shape}")
    primitives = raw.reshape(-1, 7, int(sampled_steps), 3)
    base, include_gram, time_basis = parse_task5_feature_recipe(recipe)
    parts: list[np.ndarray] = []
    if base is not None:
        cached = np.asarray(source["fmt_features"], dtype=np.float32)
        indices = fmt_feature_indices_3d(base)
        parts.append(cached[:, indices])
    if include_gram:
        parts.append(time_local_gram_dft_features_3d(
            torch.from_numpy(primitives),
            num_freq=int(gram_num_freq),
            subtract_initial=bool(gram_subtract_initial),
            normalize_initial_scale=bool(gram_normalize_initial_scale),
        ).astype(np.float32))
    if time_basis is not None:
        sample_times = None
        if time_basis == "physical":
            sample_times = task5_sample_times(
                source["integration_steps"], source["physical_dt"], sampled_steps
            )
        kinematic_input = torch.from_numpy(primitives)
        if time_basis == "physical":
            kinematic_input = kinematic_input.double()
        parts.append(pathline_velocity_gradient_dft_features_3d(
            kinematic_input,
            num_freq=int(kinematic_num_freq),
            eps=float(kinematic_pinv_rtol),
            sample_times=sample_times,
            log_compress=bool(kinematic_log_compress),
        ).astype(np.float32))
    if not parts:
        raise AssertionError(f"recipe {recipe!r} produced no feature blocks")
    result = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
    if not np.isfinite(result).all():
        raise ValueError(f"recipe {recipe!r} produced non-finite features")
    return np.ascontiguousarray(result, dtype=np.float32)
