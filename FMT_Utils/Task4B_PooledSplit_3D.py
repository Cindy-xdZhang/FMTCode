"""Instance-level train/validation/test splitting for pooled Task4-b volumes.

Every hairpin instance (``VortexId > 0``) is assigned as a whole to one split.
Ordinary vortex cubes inherit the split of the nearest hairpin cube of the same
volume (a Voronoi partition by hairpin instance).  A spatial buffer then removes
train/validation cubes whose seed lies within ``buffer_distance`` of any test
cube seed, so no training primitive support can reach a test seed.  All helpers
are label-free with respect to the four classes: they only read instance ids,
seeds and the volume geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_CODES = {name: code for code, name in enumerate(SPLIT_NAMES)}


def size_stratified_instance_split(
    instance_ids: np.ndarray,
    instance_sizes: np.ndarray,
    *,
    seed: int,
    block_pattern: tuple[str, ...] = (
        "test", "test", "validation",
        "train", "train", "train", "train", "train", "train", "train",
    ),
) -> dict[str, np.ndarray]:
    """Assign instances to splits in size-sorted blocks with a fixed pattern.

    Instances are sorted by cube count (descending).  Within each consecutive
    block of ``len(block_pattern)`` instances the pattern is randomly permuted,
    so each split receives large, medium and small instances in the same
    proportion.  A trailing partial block draws a random subset of pattern slots.
    """
    ids = np.asarray(instance_ids, dtype=np.int64)
    sizes = np.asarray(instance_sizes, dtype=np.int64)
    if ids.ndim != 1 or ids.shape != sizes.shape or len(ids) == 0:
        raise ValueError("instance ids and sizes must be equal-length 1D arrays")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("instance ids must be unique")
    for name in block_pattern:
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split in block pattern: {name}")
    rng = np.random.default_rng(int(seed))
    order = np.lexsort((ids, -sizes))  # descending size, ties by id
    pattern = np.asarray(block_pattern)
    assignment: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    block = len(pattern)
    for start in range(0, len(order), block):
        members = order[start : start + block]
        slots = rng.permutation(len(pattern))[: len(members)]
        for member, slot in zip(members, slots):
            assignment[str(pattern[slot])].append(int(ids[member]))
    return {name: np.asarray(sorted(values), dtype=np.int64) for name, values in assignment.items()}


def _wrapped_copies(points: np.ndarray, periods_xy: tuple[float, float] | None) -> np.ndarray:
    """Return points plus their periodic images in x/y so seam distances are exact."""
    points = np.asarray(points, dtype=np.float64)
    if periods_xy is None:
        return points
    copies = []
    for shift_x in (-1, 0, 1):
        for shift_y in (-1, 0, 1):
            offset = np.asarray([shift_x * periods_xy[0], shift_y * periods_xy[1], 0.0])
            copies.append(points + offset[None, :])
    return np.concatenate(copies, axis=0)


def nearest_instance_ids(
    query_seeds: np.ndarray,
    hairpin_seeds: np.ndarray,
    hairpin_ids: np.ndarray,
    *,
    periods_xy: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest hairpin cube per query seed; returns (instance id, distance)."""
    hairpin_seeds = np.asarray(hairpin_seeds, dtype=np.float64)
    hairpin_ids = np.asarray(hairpin_ids, dtype=np.int64)
    if len(hairpin_seeds) == 0:
        raise ValueError("no hairpin seeds to define the Voronoi partition")
    reference = _wrapped_copies(hairpin_seeds, periods_xy)
    reference_ids = np.tile(hairpin_ids, len(reference) // len(hairpin_ids))
    tree = cKDTree(reference)
    distance, index = tree.query(np.asarray(query_seeds, dtype=np.float64), k=1)
    return reference_ids[index], np.asarray(distance, dtype=np.float64)


def buffer_mask_against_test(
    seeds: np.ndarray,
    split_codes: np.ndarray,
    *,
    buffer_distance: float,
    periods_xy: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (keep mask, distance to nearest test seed).

    Train/validation cubes closer than ``buffer_distance`` to any test cube seed
    are dropped.  Test cubes are always kept.
    """
    seeds = np.asarray(seeds, dtype=np.float64)
    codes = np.asarray(split_codes)
    test = codes == SPLIT_CODES["test"]
    if not test.any():
        raise ValueError("no test cubes to buffer against")
    tree = cKDTree(_wrapped_copies(seeds[test], periods_xy))
    distance, _ = tree.query(seeds, k=1)
    keep = test | (distance >= float(buffer_distance))
    return keep, np.asarray(distance, dtype=np.float64)


@dataclass(frozen=True)
class VolumeSplitResult:
    split_codes: np.ndarray
    keep_mask: np.ndarray
    distance_to_test: np.ndarray
    instance_assignment: dict[str, np.ndarray]
    nearest_instance: np.ndarray
    summary: dict


def split_volume(
    seeds: np.ndarray,
    vortex_ids: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    buffer_distance: float,
    periods_xy: tuple[float, float] | None,
) -> VolumeSplitResult:
    """Instance split + Voronoi assignment of ordinary cubes + test buffer."""
    seeds = np.asarray(seeds, dtype=np.float64)
    ids = np.asarray(vortex_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if not (len(seeds) == len(ids) == len(labels)):
        raise ValueError("seeds, vortex ids and labels must have equal length")
    hairpin = ids > 0
    if not np.array_equal(hairpin, labels >= 2):
        raise ValueError("hairpin support must coincide with head/limb labels")
    instance_ids, sizes = np.unique(ids[hairpin], return_counts=True)
    assignment = size_stratified_instance_split(instance_ids, sizes, seed=seed)
    instance_split = np.full(int(instance_ids.max()) + 1, -1, dtype=np.int8)
    for name, members in assignment.items():
        instance_split[members] = SPLIT_CODES[name]
    nearest, _ = nearest_instance_ids(
        seeds, seeds[hairpin], ids[hairpin], periods_xy=periods_xy
    )
    owner = np.where(hairpin, ids, nearest)
    split_codes = instance_split[owner]
    if (split_codes < 0).any():
        raise RuntimeError("some cubes received no split")
    keep, distance = buffer_mask_against_test(
        seeds, split_codes, buffer_distance=buffer_distance, periods_xy=periods_xy
    )
    summary = {
        "instance_count": int(len(instance_ids)),
        "instances_per_split": {name: int(len(v)) for name, v in assignment.items()},
        "hairpin_cubes_per_split_before_buffer": {
            name: int(np.count_nonzero(hairpin & (split_codes == code)))
            for name, code in SPLIT_CODES.items()
        },
        "hairpin_cubes_per_split_after_buffer": {
            name: int(np.count_nonzero(hairpin & keep & (split_codes == code)))
            for name, code in SPLIT_CODES.items()
        },
        "ordinary_cubes_per_split_before_buffer": {
            name: int(np.count_nonzero(~hairpin & (split_codes == code)))
            for name, code in SPLIT_CODES.items()
        },
        "ordinary_cubes_per_split_after_buffer": {
            name: int(np.count_nonzero(~hairpin & keep & (split_codes == code)))
            for name, code in SPLIT_CODES.items()
        },
        "buffer_distance": float(buffer_distance),
        "minimum_kept_nontest_distance_to_test": float(
            distance[keep & (split_codes != SPLIT_CODES["test"])].min()
        ) if np.any(keep & (split_codes != SPLIT_CODES["test"])) else None,
    }
    return VolumeSplitResult(
        split_codes=split_codes.astype(np.int8),
        keep_mask=keep,
        distance_to_test=distance,
        instance_assignment=assignment,
        nearest_instance=owner.astype(np.int64),
        summary=summary,
    )
