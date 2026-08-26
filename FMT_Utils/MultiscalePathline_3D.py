"""Exact variable-scale 3D pathline-cross integration.

Each spatial seed is assigned one scale tuple.  The tuple controls the
centre-to-neighbour distance, numerical integration step, and integration
step count.  Every resulting primitive is nevertheless resampled to the same
``[7, L, 4]`` tensor so mixed scales can share a neural-network batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from FMT_Utils.FMT_3D_pipeline import integrate_cross_primitives_3d


@dataclass(frozen=True)
class PathlineScale3D:
    """One discrete integration scale expressed in source-grid units."""

    name: str
    offset_grid_scale: float
    dt_scale: float
    integration_steps: int

    @property
    def horizon_in_source_frames(self) -> float:
        return float(self.dt_scale) * int(self.integration_steps)


def parse_scale_table(rows, sampled_steps: int) -> list[PathlineScale3D]:
    """Validate and freeze a YAML scale table."""
    scales = [
        PathlineScale3D(
            name=str(row["name"]),
            offset_grid_scale=float(row["offset_grid_scale"]),
            dt_scale=float(row["dt_scale"]),
            integration_steps=int(row["integration_steps"]),
        )
        for row in rows
    ]
    if not scales:
        raise ValueError("a multiscale table must contain at least one scale")
    names = [scale.name for scale in scales]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate multiscale names: {names}")
    sampled_steps = int(sampled_steps)
    for scale in scales:
        if not np.isfinite(scale.offset_grid_scale) or scale.offset_grid_scale <= 0:
            raise ValueError(f"{scale.name}: offset_grid_scale must be positive")
        if not np.isfinite(scale.dt_scale) or scale.dt_scale <= 0:
            raise ValueError(f"{scale.name}: dt_scale must be positive")
        if scale.integration_steps < sampled_steps - 1:
            raise ValueError(
                f"{scale.name}: integration_steps={scale.integration_steps} "
                f"cannot provide L={sampled_steps} unique samples"
            )
    return scales


def balanced_scale_assignment(sample_count: int, scale_count: int, seed: int) -> np.ndarray:
    """Assign scales independently of seed position with count imbalance <= 1."""
    sample_count, scale_count = int(sample_count), int(scale_count)
    if sample_count < 0 or scale_count <= 0:
        raise ValueError("sample_count must be non-negative and scale_count positive")
    if sample_count == 0:
        return np.empty(0, dtype=np.int16)
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(sample_count)
    assignment = np.empty(sample_count, dtype=np.int16)
    assignment[permutation] = np.arange(sample_count, dtype=np.int64) % scale_count
    counts = np.bincount(assignment, minlength=scale_count)
    if int(counts.max() - counts.min()) > 1:
        raise AssertionError("balanced scale assignment failed")
    return assignment


def positive_grid_intervals(vector_field) -> np.ndarray:
    spacing = np.asarray(vector_field.gridInterval, dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"expected three positive grid intervals, got {spacing}")
    return spacing


def integrate_multiscale_primitives_3d(
    vector_field,
    seeds_xyz,
    seed_time: float,
    scales: list[PathlineScale3D],
    scale_assignment,
    sampled_steps: int,
    *,
    offset_mode: str = "min",
    method: str = "RK4",
    chunk_size: int = 2048,
):
    """Integrate one exact scale per seed and restore original seed ordering.

    Returns only primitives whose seven lines are complete and spatially
    valid.  ``valid_mask`` and ``line_lengths`` retain the original seed-grid
    indexing, while every other returned array is indexed by valid seed.
    """
    seeds = np.asarray(seeds_xyz, dtype=np.float64)
    assignment = np.asarray(scale_assignment, dtype=np.int64)
    if seeds.ndim != 2 or seeds.shape[1] != 3:
        raise ValueError(f"seeds_xyz must be [N,3], got {seeds.shape}")
    if assignment.shape != (len(seeds),):
        raise ValueError("scale_assignment must contain one id per seed")
    if assignment.size and (assignment.min() < 0 or assignment.max() >= len(scales)):
        raise ValueError("scale_assignment contains an unknown scale id")

    spacing = positive_grid_intervals(vector_field)
    if offset_mode == "min":
        offset_base = float(spacing.min())
    elif offset_mode == "geometric_mean":
        offset_base = float(np.prod(spacing) ** (1.0 / 3.0))
    elif offset_mode == "max":
        offset_base = float(spacing.max())
    else:
        raise ValueError("offset_mode must be min, geometric_mean, or max")

    sampled_steps = int(sampled_steps)
    primitives_all = np.zeros((len(seeds), 7, sampled_steps, 4), dtype=np.float64)
    lengths_all = np.zeros((len(seeds), 7), dtype=np.int64)
    valid_all = np.zeros(len(seeds), dtype=bool)
    physical_offsets = np.zeros(len(seeds), dtype=np.float64)
    physical_dts = np.zeros(len(seeds), dtype=np.float64)
    horizons = np.zeros(len(seeds), dtype=np.float64)

    for scale_id, scale in enumerate(scales):
        selected = np.flatnonzero(assignment == scale_id)
        if not len(selected):
            continue
        offset = offset_base * scale.offset_grid_scale
        dt = float(vector_field.timeInterval) * scale.dt_scale
        primitives, valid_local, lengths = integrate_cross_primitives_3d(
            vector_field,
            seeds[selected],
            float(seed_time),
            dt,
            scale.integration_steps,
            sampled_steps,
            offset,
            method=method,
            chunk_size=int(chunk_size),
        )
        lengths_all[selected] = lengths
        valid_indices = selected[valid_local]
        primitives_all[valid_indices] = primitives
        valid_all[valid_indices] = True
        physical_offsets[selected] = offset
        physical_dts[selected] = dt
        horizons[selected] = dt * scale.integration_steps

    return {
        "primitives": primitives_all[valid_all],
        "valid_mask": valid_all,
        "line_lengths": lengths_all,
        "scale_id": assignment[valid_all].astype(np.int16),
        "offset_grid_scale": np.asarray(
            [scales[index].offset_grid_scale for index in assignment[valid_all]],
            dtype=np.float32,
        ),
        "dt_scale": np.asarray(
            [scales[index].dt_scale for index in assignment[valid_all]],
            dtype=np.float32,
        ),
        "integration_steps": np.asarray(
            [scales[index].integration_steps for index in assignment[valid_all]],
            dtype=np.int16,
        ),
        "primitive_offset": physical_offsets[valid_all].astype(np.float32),
        "physical_dt": physical_dts[valid_all].astype(np.float32),
        "integration_horizon": horizons[valid_all].astype(np.float32),
    }


def maximum_horizon_in_source_frames(scales: list[PathlineScale3D]) -> float:
    return max(scale.horizon_in_source_frames for scale in scales)


def maximum_offset_grid_scale(scales: list[PathlineScale3D]) -> float:
    return max(scale.offset_grid_scale for scale in scales)
