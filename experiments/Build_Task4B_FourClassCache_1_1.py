"""Build a leakage-aware channel cache for Task4-b clustering and training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.Task4A_StreamlineClustering_3D import (
    ChannelVelocityField3D,
    integrate_bidirectional_cross_primitives,
)
from FMT_Utils.Task4B_ProxyLabels_3D import CLASS_NAMES


DEFAULT_CONFIG = Path("config/Other_Task4B_FMTFourClassClustering_1.2.yaml")
SPLIT_NAMES = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(candidates: list[str], override: str | None) -> Path:
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"no channel flow found among {candidates}")


def _assign_split_codes(
    seeds_xyz: np.ndarray,
    vortex_ids: np.ndarray,
    label_data,
    split_spec: dict,
    period_x: float,
    domain_min_x: float,
) -> np.ndarray:
    """Use complete hairpin IDs and the same buffered x slabs for ordinary cubes."""

    codes = np.full(len(seeds_xyz), -1, dtype=np.int8)
    for code, name in enumerate(SPLIT_NAMES):
        ids = np.asarray(label_data[f"{name}_vortex_ids"], dtype=np.int32)
        codes[np.isin(vortex_ids, ids)] = code
    ordinary = vortex_ids <= 0
    x_fraction = np.mod(seeds_xyz[:, 0] - float(domain_min_x), period_x) / period_x
    intervals = split_spec.get(
        "x_intervals", split_spec.get("ordinary_x_intervals")
    )
    if intervals is None:
        raise KeyError("splits requires x_intervals")
    for code, name in enumerate(SPLIT_NAMES):
        lower, upper = (float(value) for value in intervals[name])
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(f"invalid ordinary x interval for {name}")
        codes[ordinary & (x_fraction >= lower) & (x_fraction < upper)] = code
    return codes


def _sample_indices(
    labels: np.ndarray,
    split_codes: np.ndarray,
    sampling_spec: dict,
    *,
    seed: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(int(seed))
    oversample = float(sampling_spec["ordinary_oversample_factor"])
    selected = []
    rows = []
    for split_code, split_name in enumerate(SPLIT_NAMES):
        for class_id, class_name in enumerate(CLASS_NAMES):
            candidates = np.flatnonzero(
                (split_codes == split_code) & (labels == class_id)
            )
            configured = sampling_spec["max_per_class_per_split"][class_name][
                split_name
            ]
            target = None if configured is None else int(configured)
            draw_count = len(candidates)
            if target is not None:
                draw_count = min(
                    len(candidates), max(target, int(np.ceil(target * oversample)))
                )
            if draw_count < len(candidates):
                drawn = np.sort(rng.choice(candidates, size=draw_count, replace=False))
            else:
                drawn = candidates
            selected.append(drawn)
            rows.append(
                {
                    "split": split_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "available_count": int(len(candidates)),
                    "target_after_validity": target,
                    "integrated_count": int(len(drawn)),
                }
            )
    return np.sort(np.concatenate(selected)), {"rows": rows}


def _trim_valid_to_targets(
    source_indices: np.ndarray,
    valid: np.ndarray,
    labels: np.ndarray,
    split_codes: np.ndarray,
    sampling_spec: dict,
    *,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(int(seed) + 1)
    keep_local = []
    rows = []
    for split_code, split_name in enumerate(SPLIT_NAMES):
        for class_id, class_name in enumerate(CLASS_NAMES):
            member = np.flatnonzero(
                valid
                & (split_codes[source_indices] == split_code)
                & (labels[source_indices] == class_id)
            )
            configured = sampling_spec["max_per_class_per_split"][class_name][
                split_name
            ]
            target = None if configured is None else int(configured)
            if target is not None and len(member) > target:
                member = np.sort(rng.choice(member, size=target, replace=False))
            minimum = int(sampling_spec["minimum_per_class_per_split"])
            if len(member) < minimum:
                raise RuntimeError(
                    f"{split_name}/{class_name}: only {len(member)} valid cubes, "
                    f"minimum is {minimum}"
                )
            keep_local.append(member)
            rows.append(
                {
                    "split": split_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "valid_before_trim": int(
                        np.count_nonzero(
                            valid
                            & (split_codes[source_indices] == split_code)
                            & (labels[source_indices] == class_id)
                        )
                    ),
                    "cached_count": int(len(member)),
                }
            )
    return np.sort(np.concatenate(keep_local)), rows


def build(
    config_path: str | Path,
    *,
    flow_override: str | None = None,
    output_override: str | None = None,
) -> Path:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cache_spec = spec["cache"]
    label_path = Path(spec["input"]["proxy_label_result"])
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    output_path = Path(output_override or cache_spec["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output_path.parent / "cache_config_snapshot.yaml"
    if snapshot.exists():
        old = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        if json.loads(json.dumps(old, sort_keys=True)) != json.loads(
            json.dumps(spec, sort_keys=True)
        ):
            raise RuntimeError("cache configuration changed; use a new experiment version")
    snapshot.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    flow_path = _resolve_path(
        spec["input"]["flow_path_candidates"], flow_override
    )
    field, flow_metadata = ChannelVelocityField3D.from_vtk(flow_path)
    with np.load(label_path) as label_data:
        seeds = np.asarray(label_data["candidate_seeds_xyz"], dtype=np.float32)
        labels = np.asarray(label_data["candidate_labels"], dtype=np.int8)
        vortex_ids = np.asarray(label_data["candidate_vortex_ids"], dtype=np.int32)
        indices_xyz = np.asarray(label_data["candidate_indices_xyz"], dtype=np.int32)
        resolution_xyz = np.asarray(label_data["resolution_xyz"], dtype=np.int32)
        domain_min_xyz = np.asarray(label_data["domain_min_xyz"], dtype=np.float64)
        domain_max_xyz = np.asarray(label_data["domain_max_xyz"], dtype=np.float64)
        split_codes = _assign_split_codes(
            seeds,
            vortex_ids,
            label_data,
            spec["splits"],
            field.periods_xy[0],
            domain_min_xyz[0],
        )

    selected_source, selection_metadata = _sample_indices(
        labels,
        split_codes,
        cache_spec["sampling"],
        seed=int(spec["seed"]),
    )
    streamline_spec = spec["streamlines"]
    voxel_size = (domain_max_xyz - domain_min_xyz) / resolution_xyz
    spatial_step = float(
        streamline_spec["spatial_step_mean_voxel_scale"] * np.mean(voxel_size)
    )
    offset = float(
        streamline_spec["offset_min_voxel_scale"] * np.min(voxel_size)
    )
    intervals = spec["splits"].get(
        "x_intervals", spec["splits"].get("ordinary_x_intervals")
    )
    ordered = sorted(
        (float(bounds[0]), float(bounds[1]), name)
        for name, bounds in intervals.items()
    )
    cyclic_gaps = [
        ordered[index + 1][0] - ordered[index][1]
        for index in range(len(ordered) - 1)
    ] + [1.0 - ordered[-1][1] + ordered[0][0]]
    minimum_gap_x = min(cyclic_gaps) * float(field.periods_xy[0])
    primitive_radius = (
        int(streamline_spec["steps_per_direction"]) * spatial_step + offset
    )
    if minimum_gap_x <= 2.0 * primitive_radius:
        raise RuntimeError(
            "split x buffer is too narrow for non-overlapping primitives: "
            f"minimum_gap={minimum_gap_x:.9g}, "
            f"required>{2.0 * primitive_radius:.9g}"
        )
    primitives, valid = integrate_bidirectional_cross_primitives(
        field,
        seeds[selected_source],
        spatial_step=spatial_step,
        steps_per_direction=int(streamline_spec["steps_per_direction"]),
        offset=offset,
        minimum_speed=float(streamline_spec["minimum_speed"]),
        chunk_size=int(streamline_spec["chunk_size"]),
        unit_velocity=True,
    )
    keep_local, final_rows = _trim_valid_to_targets(
        selected_source,
        valid,
        labels,
        split_codes,
        cache_spec["sampling"],
        seed=int(spec["seed"]),
    )
    source_indices = selected_source[keep_local]
    primitives = primitives[keep_local]
    encoder = spec["encoder"]
    fmt_features = pathline_dft_features_3d(
        torch.from_numpy(primitives),
        num_freq=int(encoder["num_freq"]),
        neighbor_weight=1.0,
        neighbor_scale=float(encoder["neighbor_scale"]),
        neighbor_pool=str(encoder["neighbor_pool"]),
        mode=str(encoder["mode"]),
        include_chirality=bool(encoder["include_chirality"]),
    ).astype(np.float32)
    raw_features = (
        primitives - primitives[:, :1, :1, :]
    ).reshape(len(primitives), -1).astype(np.float32)
    cached_labels = labels[source_indices]
    cached_splits = split_codes[source_indices]
    if not np.isfinite(raw_features).all() or not np.isfinite(fmt_features).all():
        raise RuntimeError("valid cache contains non-finite Raw/FMT features")

    metadata = {
        "experiment": spec["experiment"],
        "config": str(config_path.resolve()),
        "proxy_label_result": str(label_path.resolve()),
        "proxy_label_sha256": _sha256(label_path),
        "flow": flow_metadata,
        "streamlines": {
            "method": streamline_spec["method"],
            "spatial_step": spatial_step,
            "offset": offset,
            "steps_per_direction": int(streamline_spec["steps_per_direction"]),
            "sampled_steps": int(primitives.shape[2]),
            "integrated_count": int(len(selected_source)),
            "invalid_count": int(np.count_nonzero(~valid)),
            "cached_count": int(len(source_indices)),
        },
        "encoder": encoder,
        "split_policy": spec["splits"],
        "split_nonoverlap_certificate": {
            "minimum_cyclic_x_gap": minimum_gap_x,
            "primitive_support_radius_upper_bound": primitive_radius,
            "required_minimum_gap": 2.0 * primitive_radius,
            "certified": True,
            "reason": (
                "all complete hairpin IDs and ordinary cubes lie inside the same "
                "x slabs; the minimum cyclic slab gap exceeds two primitive radii"
            ),
        },
        "initial_selection": selection_metadata["rows"],
        "final_counts": final_rows,
    }
    np.savez_compressed(
        output_path,
        raw_features=raw_features,
        fmt_features=fmt_features,
        labels=cached_labels,
        split_codes=cached_splits,
        seeds_xyz=seeds[source_indices],
        vortex_ids=vortex_ids[source_indices],
        voxel_indices_xyz=indices_xyz[source_indices],
        source_candidate_indices=source_indices,
        center_streamlines=primitives[:, 0],
        resolution_xyz=resolution_xyz,
        domain_min_xyz=domain_min_xyz,
        domain_max_xyz=domain_max_xyz,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        "cache_path": str(output_path.resolve()),
        "shape": {
            "raw_features": list(raw_features.shape),
            "fmt_features": list(fmt_features.shape),
        },
        "split_class_counts": final_rows,
        "metadata": metadata,
    }
    (output_path.parent / "cache_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["shape"], indent=2), flush=True)
    print(json.dumps(final_rows, indent=2), flush=True)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--flow")
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build(args.config, flow_override=args.flow, output_override=args.output)
