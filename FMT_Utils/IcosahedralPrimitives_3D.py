"""13-line icosahedral primitives: a centre pathline and twelve neighbours.

The octahedral pipeline (`integrate_cross_primitives_3d`) seeds six neighbours
along +-x, +-y, +-z.  This seeds twelve along the icosahedron vertices, which is
the arrangement whose symmetry group acts on the neighbour channels by
permutation and which covers SO(3) with a mean gap of 29.5 degrees against the
octahedron's 40.8 (measured in `tests/test_icosahedral_group_3d.py`).

The twelve unit vertices have maximum absolute component phi/sqrt(1+phi^2) ~ 0.851,
so a star of radius `offset` reaches *less* far along any axis than the
octahedral star of the same radius; the existing boundary margin is therefore
still sufficient and the seeding grid is shared unchanged.
"""
from __future__ import annotations

import numpy as np

from FLowUtils.flowlineIntegral import compute_pathlines_3D_batch
from FMT_Utils.IcosahedralGroup_3D import LINE_COUNT, icosahedron_vertices


def icosahedral_offsets(offset):
    """``[13, 3]`` seed offsets: the centre, then the twelve vertices."""
    return np.concatenate((np.zeros((1, 3)), float(offset) * icosahedron_vertices()))


def integrate_icosahedral_primitives_3d(vector_field, seeds_xyz, seed_time, dt,
                                        integration_steps, sampled_steps, offset,
                                        method="RK4", chunk_size=2048):
    """Integrate a 13-line star per seed; keep only fully valid primitives.

    Returns ``(primitives [M,13,S,4], valid_mask [N], lengths [N,13])`` with the
    same validity rule as the octahedral builder: every line must reach the full
    step count and stay inside the domain.
    """
    if integration_steps < 1 or not 2 <= sampled_steps <= integration_steps + 1:
        raise ValueError("require integration_steps>=1 and 2<=sampled_steps<=steps+1")
    target_time = float(seed_time) + float(dt) * int(integration_steps)
    if not vector_field.tmin <= seed_time <= vector_field.tmax:
        raise ValueError("seed_time is outside the field time range")
    if target_time > vector_field.tmax + 1e-12:
        raise ValueError(f"integration target {target_time:g} exceeds tmax={vector_field.tmax:g}")

    offsets = icosahedral_offsets(offset)
    expanded = (np.asarray(seeds_xyz)[:, None, :] + offsets[None]).reshape(-1, 3)
    expanded = np.column_stack((expanded, np.full(len(expanded), seed_time)))
    desired_length = integration_steps + 1
    sample_index = np.linspace(0, integration_steps, sampled_steps).round().astype(np.int64)

    chunks, length_chunks = [], []
    for start in range(0, len(expanded), int(chunk_size)):
        result = compute_pathlines_3D_batch(
            vector_field, expanded[start:start + int(chunk_size)],
            min_time=float(seed_time), max_time=target_time, step_size=float(dt),
            max_iteration=int(integration_steps), method=method)
        if result is None:
            raise RuntimeError(f"unsupported 3D integration method: {method}")
        positions, lengths = result
        chunks.append(positions[:, :desired_length])
        length_chunks.append(lengths)

    positions = np.concatenate(chunks).reshape(-1, LINE_COUNT, desired_length, 4)
    lengths = np.concatenate(length_chunks).reshape(-1, LINE_COUNT)
    lower = np.asarray(vector_field.domainMinBoundary).reshape(1, 1, 1, 3)
    upper = np.asarray(vector_field.domainMaxBoundary).reshape(1, 1, 1, 3)
    xyz = positions[..., :3]
    inside = ((xyz >= lower) & (xyz <= upper)).all(axis=(1, 2, 3))
    valid = (lengths == desired_length).all(axis=1) & inside
    return positions[valid][:, :, sample_index], valid, lengths
