"""Versioned, label-free flow-map material bundles for Tasks 6/7/8."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

import netCDF4 as nc
import numpy as np
from FMT_Utils.NetCDF_window_3D import _axis_dimension, _coordinate, load_netcdf_window_3d
from FMT_Utils.FlowMapData_3D import build_window, select_windows, sha256, write_json

DEFAULT_CONFIG = "config/mainExp_Task678_FlowMap_1.1.json"


def separated_seed_grid(lo, hi, radius, maximum_side):
    """Reduce only the count on a thin axis; do not overlap Task7 regions."""
    counts = np.minimum(maximum_side, np.floor((hi - lo) / (10.1 * radius)).astype(int) + 1)
    counts = np.maximum(counts, 1)
    axes = [np.linspace(a, b, int(n)) if n > 1 else np.array([(a + b) / 2])
            for a, b, n in zip(lo, hi, counts)]
    grid = np.meshgrid(*axes, indexing="ij")
    steps = [(b - a) / (n - 1) for a, b, n in zip(lo, hi, counts) if n > 1]
    return np.stack(grid, axis=-1).reshape(-1, 3), counts.tolist(), min(steps) if steps else 0.


def make_features(data, arms):
    from FMT_Utils.FlowMapModels_3D import raw_features
    result = {}
    n = len(data["origin0"])
    for arm in arms:
        for stage in (0, 1):
            result[f"{arm}__support{stage}"] = raw_features(data[f"support{stage}"], data[f"radius{stage}"], arm)
        # This mean is computed exclusively from visible context primitives.
        result[f"{arm}__context"] = raw_features(data["context"].reshape(n * 6, 31, 32, 3),
            np.repeat(data["radius0"], 6), arm).reshape(n, 6, -1)
    return result


def provenance(config):
    return dict(config_sha256=sha256(config),
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                node=socket.gethostname(), job_id=os.getenv("SLURM_JOB_ID"),
                array_id=os.getenv("SLURM_ARRAY_TASK_ID"),
                time=datetime.datetime.now().astimezone().isoformat())


def inspect_source(spec, dataset):
    path = Path(spec["source_fields"][dataset])
    with nc.Dataset(path) as d:
        td = _axis_dimension(d, "t")
        t = _coordinate(d, td, np.arange(len(d.dimensions[td])))
        available = list(range(len(t)))
        if "selected_original_indices_json" in d.ncattrs():
            available = json.loads(d.getncattr("selected_original_indices_json"))
        if dataset in spec.get("available_frame_ranges", {}):
            lo, hi = spec["available_frame_ranges"][dataset]
            available = sorted(set(available).intersection(range(lo, hi + 1)))
        extraction = str(getattr(d, "extraction", "original source"))
        if "missing" in extraction.lower() and len(available) == len(t):
            raise ValueError("Sparse export must explicitly identify available frames")
    s = spec["sampling"]
    starts = select_windows(t, available, s["windows"], s["source_frames"],
        spec["minimum_cylinder_time"] if dataset in spec["cylinder_datasets"] else None)
    rows = []
    for ordinal, start in enumerate(starts):
        role = next(role for role, ids in spec["splits"].items() if ordinal in ids)
        rows.append(dict(ordinal=ordinal, role=role, start_index=start,
                         end_index=start + s["source_frames"] - 1,
                         time_start=float(t[start]), time_end=float(t[start + s["source_frames"] - 1])))
    return dict(dataset=dataset, source=str(path), source_bytes=path.stat().st_size,
                source_mtime_ns=path.stat().st_mtime_ns, extraction=extraction,
                available_frame_indices=available, windows=rows)


def check_present_frames(path, start, count):
    """Reject wholly missing exported frames before the legacy loader fills masks."""
    with nc.Dataset(path) as d:
        td = _axis_dimension(d, "t")
        names = next(x for x in (("u", "v", "w"), ("velocity_x", "velocity_y", "velocity_z"),
                                ("Component1", "Component2", "Component3")) if all(k in d.variables for k in x))
        for name in names:
            v = d.variables[name]
            for k in range(start, start + count):
                # A coarse spatial scan establishes existence, not validity of
                # every voxel. Physical solid masks retain the source convention.
                slices = tuple(k if axis == td else slice(None, None, max(1, size // 8))
                               for axis, size in zip(v.dimensions, v.shape))
                a = np.ma.asarray(v[slices])
                if not a.count() or not np.isfinite(a.compressed()).all():
                    raise ValueError(f"Missing/nonfinite source frame {path}/{name}/{k}")


def build(spec, config, dataset):
    s = spec["sampling"]
    if s["steps_per_segment"] != 62 or s["source_frames"] != 9 or s["context_offset_radii"] != 3.:
        raise ValueError("Changed geometry constants require a new experiment version")
    root = Path(spec["output_root"])
    source = inspect_source(spec, dataset)
    initial = {**provenance(config), **source, "status": "building"}
    write_json(root / "build" / f"{dataset}.started.json", initial)
    arrays = []
    for row in source["windows"]:
        path = root / "cache" / dataset / f"window_{row['ordinal']:02d}.npz"
        if path.exists():
            saved = json.loads(path.with_suffix(".json").read_text())
            if saved["config_sha256"] != sha256(config) or saved["cache_sha256"] != sha256(path):
                raise ValueError(f"Existing cache identity mismatch: {path}")
            arrays.append(saved)
            continue
        check_present_frames(source["source"], row["start_index"], s["source_frames"])
        field, meta = load_netcdf_window_3d(source["source"], row["start_index"],
                                          s["source_frames"], s["max_spatial_dim"])
        temporal_change = float(np.linalg.norm(np.diff(field.field.astype(np.float64), axis=0)))
        if temporal_change <= 1e-12 or not np.isfinite(field.field).all():
            raise ValueError("Source window is static or nonfinite; do not claim an unsteady test")
        radius = float(np.min(field.gridInterval)) * s["radius_grid_fraction"]
        lo = field.domainMinBoundary.astype(float) + 5.5 * radius
        hi = field.domainMaxBoundary.astype(float) - 5.5 * radius
        if np.any(lo >= hi):
            raise ValueError("Domain too narrow for separated visible and hidden regions")
        centers, grid_shape, grid_min_separation = separated_seed_grid(lo, hi, radius, s["seed_grid_side"])
        # Initial patch separation makes material reuse across spatial examples
        # observable; splits themselves already have disjoint source windows.
        if grid_min_separation <= 10 * radius:
            raise ValueError("Seed grid too dense for separated Task7 context regions")
        rng = np.random.default_rng(s["data_seed"] + row["ordinal"])
        centers = centers[rng.permutation(len(centers))[:s["max_bundles"]]]
        data, audit = build_window(field, centers, radius, field.tmax / 2, s["steps_per_segment"], s["data_seed"] + row["ordinal"])
        if len(data["origin0"]) < s["minimum_bundles"]:
            raise ValueError(f"Insufficient common material coverage: {audit}")
        field_hash = hashlib.sha256(np.ascontiguousarray(field.field).tobytes()).hexdigest()
        metadata = {**provenance(config), **row, **meta, **audit, "dataset": dataset,
                    "source_window_sha256": field_hash, "temporal_change_l2": temporal_change,
                    "grid_min_center_separation": float(grid_min_separation), "seed_grid_shape": grid_shape, "status": "PASS"}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data, metadata_json=np.asarray(json.dumps(metadata)))
        metadata["cache_sha256"] = sha256(path)
        write_json(path.with_suffix(".json"), metadata)
        arrays.append(metadata)
        print(f"BUILT {dataset}/{row['ordinal']} {audit['retained_bundles']}/{len(centers)}", flush=True)
    for row in source["windows"]:
        path = root / "cache" / dataset / f"window_{row['ordinal']:02d}.npz"
        fp = path.with_name(path.stem + "_features.npz")
        if fp.exists():
            fm = json.loads(fp.with_suffix(".json").read_text())
            if fm["cache_sha256"] != sha256(path) or fm["feature_sha256"] != sha256(fp) or fm["config_sha256"] != sha256(config):
                raise ValueError("Feature cache identity mismatch")
            continue
        with np.load(path, allow_pickle=False) as a:
            data = {k: a[k] for k in a.files if k != "metadata_json"}
        features = make_features(data, spec["arms"])
        np.savez_compressed(fp, **features)
        write_json(fp.with_suffix(".json"), {**provenance(config), "cache_sha256": sha256(path),
                   "feature_sha256": sha256(fp), "shapes": {k: list(v.shape) for k, v in features.items()}})
        print(f"FEATURES {dataset}/{row['ordinal']}", flush=True)
    write_json(root / "build" / f"{dataset}.json", {**initial, "status": "PASS", "caches": arrays,
                                                  "completed": provenance(config)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--dataset")
    p.add_argument("--index", type=int)
    p.add_argument("--inspect", action="store_true")
    args = p.parse_args()
    spec = json.loads(Path(args.config).read_text())
    if args.inspect:
        rows = [inspect_source(spec, d) for d in spec["datasets"]]
        write_json(Path(spec["output_root"]) / "source_plan.json", {**provenance(args.config), "sources": rows})
        for r in rows:
            print(json.dumps({"dataset": r["dataset"], "windows": r["windows"]}), flush=True)
    else:
        build(spec, args.config, args.dataset or spec["datasets"][args.index])


if __name__ == "__main__":
    main()
