"""Build the pooled channel+TBL cache for mainExp_Task4B_PooledInstanceSplit_3.1.

Channel cubes come from the frozen 1.2 proxy-label result and are integrated
here with the frozen channel streamline recipe.  TBL cubes come from the frozen
2.3 target cache (already integrated, byte-hash verified per chunk).  Both are
converted to per-volume dimensionless coordinates and FMT is recomputed on those
coordinates.  Whole hairpin instances are assigned to train/validation/test,
ordinary cubes inherit the split of their nearest hairpin instance, and a
one-primitive-radius buffer removes train/validation cubes near test seeds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from FMT_Utils.Task4A_StreamlineClustering_3D import (
    ChannelVelocityField3D,
    integrate_bidirectional_cross_primitives,
)
from FMT_Utils.Task4B_CrossFlow_3D import dimensionless_primitive_features
from FMT_Utils.Task4B_PooledSplit_3D import SPLIT_CODES, SPLIT_NAMES, split_volume
from FMT_Utils.Task4B_ProxyLabels_3D import CLASS_NAMES

DEFAULT_CONFIG = Path("config/mainExp_Task4B_PooledInstanceSplit_3.1.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(candidates: list[str], override: str | None, label: str) -> Path:
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"no {label} found among {candidates}")


def _snapshot(spec: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "cache_config_snapshot.yaml"
    if snapshot.exists():
        previous = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        if json.dumps(previous, sort_keys=True) != json.dumps(spec, sort_keys=True):
            raise RuntimeError("cache configuration changed; use a new experiment version")
    snapshot.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def _sample_rows(
    labels: np.ndarray,
    split_codes: np.ndarray,
    keep: np.ndarray,
    caps: dict,
    minimum: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]]:
    """All kept hairpin cubes plus a capped random sample of kept ordinary cubes."""
    selected = []
    rows = []
    for split_name in SPLIT_NAMES:
        code = SPLIT_CODES[split_name]
        for class_id, class_name in enumerate(CLASS_NAMES):
            members = np.flatnonzero(keep & (split_codes == code) & (labels == class_id))
            cap = None if class_id >= 2 else int(caps[split_name])
            drawn = members
            if cap is not None and len(members) > cap:
                drawn = np.sort(rng.choice(members, size=cap, replace=False))
            if len(drawn) < int(minimum):
                raise RuntimeError(
                    f"{split_name}/{class_name}: only {len(drawn)} cubes after buffer, "
                    f"minimum is {minimum}"
                )
            selected.append(drawn)
            rows.append(
                {
                    "split": split_name,
                    "class_name": class_name,
                    "available_after_buffer": int(len(members)),
                    "selected": int(len(drawn)),
                }
            )
    return np.sort(np.concatenate(selected)), rows


def _certificate(result, buffer_distance: float) -> dict:
    minimum = result.summary["minimum_kept_nontest_distance_to_test"]
    return {
        "buffer_distance": float(buffer_distance),
        "minimum_kept_nontest_distance_to_test": minimum,
        "certified": bool(minimum is not None and minimum >= buffer_distance),
        "rule": "no kept train/validation seed lies within one primitive support radius of a test seed",
    }


def _build_channel(spec: dict, volume: dict, rng: np.random.Generator, *, flow_override, smoke: bool):
    label_path = Path(volume["proxy_label_result"])
    label_sha = _sha256(label_path)
    if label_sha != volume["proxy_label_sha256"]:
        raise RuntimeError("channel proxy-label SHA-256 differs from config")
    with np.load(label_path) as data:
        seeds = np.asarray(data["candidate_seeds_xyz"], dtype=np.float32)
        labels = np.asarray(data["candidate_labels"], dtype=np.int8)
        ids = np.asarray(data["candidate_vortex_ids"], dtype=np.int32)
        indices = np.asarray(data["candidate_indices_xyz"], dtype=np.int32)
        resolution = np.asarray(data["resolution_xyz"], dtype=np.int64)
        domain_min = np.asarray(data["domain_min_xyz"], dtype=np.float64)
        domain_max = np.asarray(data["domain_max_xyz"], dtype=np.float64)
    if np.bincount(labels, minlength=4).tolist() != list(volume["expected_candidate_class_counts"]):
        raise RuntimeError("channel candidate class counts differ from config")
    if len(np.unique(ids[ids > 0])) != int(volume["expected_instance_count"]):
        raise RuntimeError("channel hairpin instance count differs from config")

    flow_path = _resolve(volume["flow_path_candidates"], flow_override, "channel flow")
    flow_sha = _sha256(flow_path)
    if flow_sha != volume["flow_sha256"]:
        raise RuntimeError("channel flow SHA-256 differs from config")
    field, flow_meta = ChannelVelocityField3D.from_vtk(flow_path)
    stream = volume["streamlines"]
    voxel = (domain_max - domain_min) / resolution
    spatial_step = float(stream["spatial_step_mean_voxel_scale"] * np.mean(voxel))
    offset = float(stream["offset_min_voxel_scale"] * np.min(voxel))
    for name, value, expected in (
        ("spatial_step", spatial_step, stream["expected_spatial_step"]),
        ("offset", offset, stream["expected_offset"]),
    ):
        if not np.isclose(value, float(expected), rtol=1e-9, atol=0.0):
            raise RuntimeError(f"channel {name} {value!r} != expected {expected!r}")
    radius = int(stream["steps_per_direction"]) * spatial_step + offset
    buffer_distance = float(spec["split"]["buffer_radius_multiplier"]) * radius
    periods = (float(field.periods_xy[0]), float(field.periods_xy[1]))

    result = split_volume(
        seeds, ids, labels, seed=int(spec["seed"]), buffer_distance=buffer_distance, periods_xy=periods
    )
    caps = dict(spec["split"]["ordinary_max_per_class_per_split_per_volume"])
    minimum = int(spec["split"]["minimum_per_class_per_split_per_volume"])
    if smoke:
        caps = {k: max(minimum, int(v) // 40) for k, v in caps.items()}
    selected, sampling_rows = _sample_rows(labels, result.split_codes, result.keep_mask, caps, minimum, rng)

    primitives, valid = integrate_bidirectional_cross_primitives(
        field,
        seeds[selected],
        spatial_step=spatial_step,
        steps_per_direction=int(stream["steps_per_direction"]),
        offset=offset,
        minimum_speed=float(stream["minimum_speed"]),
        chunk_size=int(stream["chunk_size"]),
        unit_velocity=True,
    )
    kept = selected[valid]
    primitives = primitives[valid]
    encoder = spec["encoder"]
    raw, fmt = dimensionless_primitive_features(
        primitives,
        spatial_step,
        num_freq=int(encoder["num_freq"]),
        neighbor_scale=float(encoder["neighbor_scale"]),
        neighbor_pool=str(encoder["neighbor_pool"]),
        mode=str(encoder["mode"]),
        include_chirality=bool(encoder["include_chirality"]),
    )
    arrays = {
        "raw": raw,
        "fmt": fmt,
        "labels": labels[kept],
        "split_codes": result.split_codes[kept],
        "vortex_ids": ids[kept],
        "seeds": seeds[kept],
        "voxel_indices": indices[kept],
        "source_indices": kept.astype(np.int64),
        "distance_to_test": result.distance_to_test[kept].astype(np.float32),
        "nearest_instance": result.nearest_instance[kept].astype(np.int32),
    }
    meta = {
        "proxy_label_result": str(label_path.resolve()),
        "proxy_label_sha256": label_sha,
        "flow_path": str(flow_path),
        "flow_sha256": flow_sha,
        "flow_metadata": flow_meta,
        "grid": {
            "resolution_xyz": resolution.tolist(),
            "domain_min_xyz": domain_min.tolist(),
            "domain_max_xyz": domain_max.tolist(),
        },
        "periods_xy": list(periods),
        "streamlines": {
            **{k: v for k, v in stream.items() if not k.startswith("expected_")},
            "spatial_step": spatial_step,
            "offset": offset,
            "sampled_steps": int(primitives.shape[2]),
            "primitive_support_radius": radius,
            "integrated_count": int(len(selected)),
            "invalid_count": int(np.count_nonzero(~valid)),
            "cached_count": int(len(kept)),
        },
        "split_summary": result.summary,
        "instance_assignment": {k: v.tolist() for k, v in result.instance_assignment.items()},
        "buffer_certificate": _certificate(result, buffer_distance),
        "sampling": sampling_rows,
    }
    return arrays, meta


def _build_tbl(spec: dict, volume: dict, rng: np.random.Generator, *, manifest_override, smoke: bool):
    manifest_path = _resolve(volume["target_cache_manifest_candidates"], manifest_override, "TBL manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stream = manifest["streamlines"]
    spatial_step = float(stream["spatial_step"])
    offset = float(stream["offset"])
    for name, value, expected in (
        ("spatial_step", spatial_step, volume["expected_spatial_step"]),
        ("offset", offset, volume["expected_offset"]),
    ):
        if not np.isclose(value, float(expected), rtol=1e-12, atol=0.0):
            raise RuntimeError(f"TBL {name} {value!r} != expected {expected!r}")
    radius = int(stream["steps_per_direction"]) * spatial_step + offset
    buffer_distance = float(spec["split"]["buffer_radius_multiplier"]) * radius

    chunk_paths = []
    seeds, labels, ids, indices, source = [], [], [], [], []
    for entry in manifest["chunks"]:
        path = (manifest_path.parent / entry["path_relative_to_manifest"]).resolve()
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"TBL chunk SHA-256 differs: {path}")
        chunk_paths.append(path)
        with np.load(path) as data:
            if len(data["labels"]) != int(entry["row_count"]):
                raise RuntimeError(f"TBL chunk row count differs: {path}")
            seeds.append(np.asarray(data["seeds_xyz"], dtype=np.float32))
            labels.append(np.asarray(data["labels"], dtype=np.int8))
            ids.append(np.asarray(data["vortex_ids"], dtype=np.int32))
            indices.append(np.asarray(data["voxel_indices_xyz"], dtype=np.int32))
            source.append(np.asarray(data["source_candidate_indices"], dtype=np.int64))
    offsets = np.cumsum([0] + [len(x) for x in labels])
    seeds = np.concatenate(seeds)
    labels = np.concatenate(labels)
    ids = np.concatenate(ids)
    indices = np.concatenate(indices)
    source = np.concatenate(source)
    if np.bincount(labels, minlength=4).tolist() != list(volume["expected_valid_class_counts"]):
        raise RuntimeError("TBL valid class counts differ from config")
    if len(np.unique(ids[ids > 0])) != int(volume["expected_instance_count"]):
        raise RuntimeError("TBL hairpin instance count differs from config")

    result = split_volume(
        seeds, ids, labels, seed=int(spec["seed"]) + 1, buffer_distance=buffer_distance, periods_xy=None
    )
    caps = dict(spec["split"]["ordinary_max_per_class_per_split_per_volume"])
    minimum = int(spec["split"]["minimum_per_class_per_split_per_volume"])
    if smoke:
        caps = {k: max(minimum, int(v) // 40) for k, v in caps.items()}
    selected, sampling_rows = _sample_rows(labels, result.split_codes, result.keep_mask, caps, minimum, rng)

    sampled_steps = int(stream["sampled_steps"])
    raw_rows = np.empty((len(selected), 7 * sampled_steps * 3), dtype=np.float32)
    for chunk_index, path in enumerate(chunk_paths):
        low, high = offsets[chunk_index], offsets[chunk_index + 1]
        local = selected[(selected >= low) & (selected < high)]
        if len(local) == 0:
            continue
        with np.load(path) as data:
            raw_rows[np.searchsorted(selected, local)] = np.asarray(
                data["raw_features"], dtype=np.float32
            )[local - low]
    encoder = spec["encoder"]
    raw, fmt = dimensionless_primitive_features(
        raw_rows.reshape(-1, 7, sampled_steps, 3),
        spatial_step,
        num_freq=int(encoder["num_freq"]),
        neighbor_scale=float(encoder["neighbor_scale"]),
        neighbor_pool=str(encoder["neighbor_pool"]),
        mode=str(encoder["mode"]),
        include_chirality=bool(encoder["include_chirality"]),
    )
    arrays = {
        "raw": raw,
        "fmt": fmt,
        "labels": labels[selected],
        "split_codes": result.split_codes[selected],
        "vortex_ids": ids[selected],
        "seeds": seeds[selected],
        "voxel_indices": indices[selected],
        "source_indices": source[selected],
        "distance_to_test": result.distance_to_test[selected].astype(np.float32),
        "nearest_instance": result.nearest_instance[selected].astype(np.int32),
    }
    meta = {
        "target_cache_manifest": str(manifest_path),
        "target_cache_manifest_sha256": _sha256(manifest_path),
        "chunk_count": len(chunk_paths),
        "grid": manifest["target_grid"],
        "periods_xy": None,
        "streamlines": {
            **stream,
            "primitive_support_radius": radius,
            "cached_count": int(len(selected)),
        },
        "split_summary": result.summary,
        "instance_assignment": {k: v.tolist() for k, v in result.instance_assignment.items()},
        "buffer_certificate": _certificate(result, buffer_distance),
        "sampling": sampling_rows,
    }
    return arrays, meta


def build(
    config_path: str | Path,
    *,
    flow_override: str | None = None,
    manifest_override: str | None = None,
    output_override: str | None = None,
    smoke: bool = False,
) -> Path:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if smoke:
        spec = copy.deepcopy(spec)
        spec["experiment"] = spec["experiment"] + "_smoke"
        spec["cache"]["path"] = "outputs/mainExp_Task4B_PooledInstanceSplit_3.1/smoke/cache/task4b_pooled_cache.npz"
        spec["runtime_mode"] = "smoke_not_research_result"
    cache_path = Path(output_override or spec["cache"]["path"])
    _snapshot(spec, cache_path.parent)
    rng = np.random.default_rng(int(spec["seed"]) + 100)

    channel, channel_meta = _build_channel(
        spec, spec["volumes"]["channel"], rng, flow_override=flow_override, smoke=smoke
    )
    tbl, tbl_meta = _build_tbl(
        spec, spec["volumes"]["tbl"], rng, manifest_override=manifest_override, smoke=smoke
    )
    volume_codes = np.concatenate(
        [
            np.full(len(channel["labels"]), int(spec["volumes"]["channel"]["code"]), dtype=np.int8),
            np.full(len(tbl["labels"]), int(spec["volumes"]["tbl"]["code"]), dtype=np.int8),
        ]
    )
    merged = {key: np.concatenate([channel[key], tbl[key]]) for key in channel}
    if not (np.isfinite(merged["raw"]).all() and np.isfinite(merged["fmt"]).all()):
        raise RuntimeError("pooled cache contains non-finite features")
    if merged["fmt"].shape[1] != int(spec["encoder"]["expected_feature_dim"]):
        raise RuntimeError("FMT width differs from config")

    metadata = {
        "experiment": spec["experiment"],
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "encoder": spec["encoder"],
        "representation": spec["representation"],
        "streamlines": {"sampled_steps": int(spec["representation"]["sampled_steps"])},
        "split_policy": spec["split"],
        "preregistration": spec["preregistration"],
        "volumes": {"channel": channel_meta, "tbl": tbl_meta},
        "buffer_certificates": {
            "channel": channel_meta["buffer_certificate"],
            "tbl": tbl_meta["buffer_certificate"],
            "all_certified": bool(
                channel_meta["buffer_certificate"]["certified"]
                and tbl_meta["buffer_certificate"]["certified"]
            ),
        },
        "row_counts": {
            "channel": int(len(channel["labels"])),
            "tbl": int(len(tbl["labels"])),
            "total": int(len(merged["labels"])),
        },
    }
    split_class_counts = {
        name: {
            CLASS_NAMES[c]: int(np.count_nonzero((merged["split_codes"] == code) & (merged["labels"] == c)))
            for c in range(4)
        }
        for name, code in SPLIT_CODES.items()
    }
    metadata["split_class_counts"] = split_class_counts
    np.savez_compressed(
        cache_path,
        raw_features=merged["raw"],
        fmt_features=merged["fmt"],
        labels=merged["labels"].astype(np.int8),
        split_codes=merged["split_codes"].astype(np.int8),
        volume_codes=volume_codes,
        vortex_ids=merged["vortex_ids"].astype(np.int32),
        seeds_xyz=merged["seeds"].astype(np.float32),
        voxel_indices_xyz=merged["voxel_indices"].astype(np.int32),
        source_candidate_indices=merged["source_indices"].astype(np.int64),
        distance_to_test=merged["distance_to_test"].astype(np.float32),
        nearest_instance=merged["nearest_instance"].astype(np.int32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    summary = {
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": _sha256(cache_path),
        "shape": {"raw_features": list(merged["raw"].shape), "fmt_features": list(merged["fmt"].shape)},
        "split_class_counts": split_class_counts,
        "metadata": metadata,
    }
    (cache_path.parent / "cache_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"shape": summary["shape"], "split_class_counts": split_class_counts,
                      "buffer_certificates": metadata["buffer_certificates"],
                      "cache_sha256": summary["cache_sha256"]}, indent=2), flush=True)
    return cache_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--flow")
    parser.add_argument("--tbl-manifest")
    parser.add_argument("--output")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build(
        args.config,
        flow_override=args.flow,
        manifest_override=args.tbl_manifest,
        output_override=args.output,
        smoke=args.smoke,
    )
