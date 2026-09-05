"""Reconstruct and audit the frozen whole-field IVD reference for 3D figures.

The prediction artifacts contain labels only at pathline seed locations.  A
paper figure must not turn those sparse labels into a fake volume.  This module
reloads the exact source frame, recomputes the whole-field IVD-p95 threshold,
checks every seed label against the artifact, and extracts the actual p95
isosurface.  Cached meshes are accepted only when their source and seed-label
digests match the current artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from FLowUtils.ScalarField3d import marching_cubes_world
from FMT_Utils.FMT_3D_pipeline import compute_ivd_reference_3d
from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d


SURFACE_CACHE_SCHEMA = 1
IVD_PERCENTILE = 95.0
MAX_SPATIAL_DIM = 96


def _array_digest(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _resolve_source(root, dataset, metadata):
    configured = metadata.get("source_path")
    if configured and Path(configured).exists():
        return Path(configured).resolve()

    config_paths = (
        root / "config/mainExp_Task5_3D_1.1.yaml",
        root / "config/Verify_Task2Universality_1.1.yaml",
    )
    candidates = []
    for config_path in config_paths:
        if not config_path.exists():
            continue
        spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for item in spec.get("datasets", []):
            if str(item.get("id")) != str(dataset):
                continue
            values = item.get("paths", [item.get("path")])
            candidates.extend(Path(value) for value in values if value)
    source = next((path.resolve() for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(
            f"cannot resolve a local source for {dataset}; candidates={candidates}"
        )
    return source


def _load_reference_frame(root, dataset, metadata, source_path):
    source_index = int(metadata["source_start_index"])
    if dataset == "channel" or source_path.suffix.lower() == ".vtk":
        # Import lazily so prediction-only Ibex bundles do not require VTK.
        from Build_Task5_Multiscale_Cache import _channel_fields

        spec = yaml.safe_load(
            (root / "config/mainExp_Task5_3D_1.1.yaml").read_text(
                encoding="utf-8"
            )
        )
        field, loaded_metadata = next(
            _channel_fields(source_path, spec, [source_index], frame_count=2)
        )
    else:
        field, loaded_metadata = load_netcdf_window_3d(
            source_path,
            source_index,
            frame_count=2,
            max_spatial_dim=MAX_SPATIAL_DIM,
        )

    expected_shape = metadata.get("loaded_shape_TZYXC")
    if expected_shape is not None:
        expected_spatial = tuple(int(value) for value in expected_shape[1:])
        if tuple(field.field.shape[1:]) != expected_spatial:
            raise RuntimeError(
                f"{dataset}: reconstructed spatial shape {field.field.shape[1:]} "
                f"does not match artifact {expected_spatial}"
            )
    expected_time = metadata.get("source_time")
    if expected_time is not None and not np.isclose(
        float(loaded_metadata["source_time"]),
        float(expected_time),
        rtol=1e-7,
        atol=1e-7,
    ):
        raise RuntimeError(
            f"{dataset}: reconstructed source time "
            f"{loaded_metadata['source_time']} does not match artifact "
            f"{expected_time}"
        )
    return field, loaded_metadata


def _cache_identity(dataset, source_path, metadata, seeds, reference):
    identity = {
        "schema": SURFACE_CACHE_SCHEMA,
        "dataset": str(dataset),
        "source_path": str(source_path),
        "source_start_index": int(metadata["source_start_index"]),
        "ivd_threshold": float(metadata["ivd_threshold"]),
        "seeds_sha256": _array_digest(np.asarray(seeds, dtype=np.float32)),
        "reference_sha256": _array_digest(np.asarray(reference, dtype=bool)),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity["identity_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return identity


def _load_cached_surface(path, identity):
    if not path.exists():
        return None
    with np.load(path) as data:
        audit = json.loads(str(data["audit_json"]))
        required = (
            "schema",
            "dataset",
            "source_path",
            "source_start_index",
            "ivd_threshold",
            "seeds_sha256",
            "reference_sha256",
            "identity_sha256",
        )
        if any(audit.get(key) != identity.get(key) for key in required):
            return None
        mesh = (
            np.asarray(data["vertices"], dtype=np.float32),
            np.asarray(data["normals"], dtype=np.float32),
            np.asarray(data["faces"], dtype=np.uint32),
        )
        bounds = np.asarray(data["bounds"], dtype=np.float64)
    return {"ivd_mesh": mesh, "bounds": bounds, "ivd_surface_audit": audit}


def reconstruct_ivd_reference_surface(
    root,
    dataset,
    metadata,
    seeds,
    reference,
    cache_dir,
):
    """Return an audited whole-volume p95 mesh and its physical bounds."""
    root = Path(root).resolve()
    cache_dir = Path(cache_dir)
    seeds = np.asarray(seeds, dtype=np.float32)
    reference = np.asarray(reference, dtype=bool)
    if seeds.ndim != 2 or seeds.shape[1] != 3 or len(seeds) != len(reference):
        raise ValueError("seeds/reference must have shapes [N,3] and [N]")
    if not np.isfinite(seeds).all():
        raise ValueError("seed coordinates must be finite")
    if "ivd_threshold" not in metadata:
        raise KeyError("prediction artifact does not record ivd_threshold")

    source_path = _resolve_source(root, dataset, metadata)
    identity = _cache_identity(dataset, source_path, metadata, seeds, reference)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"{dataset}_index{int(metadata['source_start_index']):04d}_"
        f"{identity['identity_sha256'][:16]}.npz"
    )
    cached = _load_cached_surface(cache_path, identity)
    if cached is not None:
        return cached

    field, loaded_metadata = _load_reference_frame(
        root, dataset, metadata, source_path
    )
    ivd_volume, ivd_at_seeds, axes = compute_ivd_reference_3d(
        field, 0.0, seeds
    )
    finite = ivd_volume[np.isfinite(ivd_volume)]
    if finite.size == 0:
        raise RuntimeError(f"{dataset}: reconstructed IVD volume has no finite values")
    computed_threshold = float(np.percentile(finite, IVD_PERCENTILE))
    artifact_threshold = float(metadata["ivd_threshold"])
    threshold_tolerance = 1e-7 * max(1.0, abs(artifact_threshold))
    if not np.isclose(
        computed_threshold,
        artifact_threshold,
        rtol=1e-7,
        atol=threshold_tolerance,
    ):
        raise RuntimeError(
            f"{dataset}: recomputed IVD-p95 threshold {computed_threshold} "
            f"does not match artifact {artifact_threshold}"
        )

    recomputed_reference = np.asarray(
        ivd_at_seeds >= artifact_threshold, dtype=bool
    )
    mismatch_count = int(np.count_nonzero(recomputed_reference != reference))
    if mismatch_count:
        mismatch_indices = np.flatnonzero(recomputed_reference != reference)[:8]
        raise RuntimeError(
            f"{dataset}: {mismatch_count} artifact seed labels disagree with "
            f"recomputed whole-field IVD-p95; first indices="
            f"{mismatch_indices.tolist()}"
        )

    spacings = []
    for axis_name, values in zip("xyz", axes):
        values = np.asarray(values, dtype=np.float64)
        differences = np.diff(values)
        if not (np.isfinite(differences).all() and np.all(differences > 0)):
            raise RuntimeError(f"{dataset}: {axis_name} coordinates are not increasing")
        spacings.append(float(np.median(differences)))
    bounds = np.asarray(
        [[axes[0][0], axes[1][0], axes[2][0]],
         [axes[0][-1], axes[1][-1], axes[2][-1]]],
        dtype=np.float64,
    )
    mesh = marching_cubes_world(
        ivd_volume,
        artifact_threshold,
        tuple(spacings),
        bounds[0],
    )
    if mesh is None:
        raise RuntimeError(f"{dataset}: IVD-p95 produced no isosurface")

    audit = {
        **identity,
        "cache_path": str(cache_path),
        "source_time": float(loaded_metadata["source_time"]),
        "ivd_definition": "whole-field |vorticity - spatial mean vorticity|",
        "ivd_percentile": IVD_PERCENTILE,
        "artifact_threshold": artifact_threshold,
        "recomputed_threshold": computed_threshold,
        "threshold_absolute_error": abs(computed_threshold - artifact_threshold),
        "reference_seed_count": int(len(reference)),
        "reference_positive_count": int(reference.sum()),
        "recomputed_label_mismatch_count": mismatch_count,
        "ivd_volume_shape_zyx": [int(value) for value in ivd_volume.shape],
        "surface_vertex_count": int(len(mesh[0])),
        "surface_face_count": int(len(mesh[2])),
    }
    np.savez_compressed(
        cache_path,
        vertices=np.asarray(mesh[0], dtype=np.float32),
        normals=np.asarray(mesh[1], dtype=np.float32),
        faces=np.asarray(mesh[2], dtype=np.uint32),
        bounds=bounds,
        audit_json=np.asarray(json.dumps(audit, sort_keys=True)),
    )
    return {"ivd_mesh": mesh, "bounds": bounds, "ivd_surface_audit": audit}
