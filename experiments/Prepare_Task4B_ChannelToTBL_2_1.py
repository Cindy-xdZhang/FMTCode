"""Prepare the full TBL test cache for channel-to-TBL Task4-b transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import yaml

from FMT_Utils.Task4A_StreamlineClustering_3D import (
    integrate_bidirectional_cross_primitives,
)
from FMT_Utils.Task4B_CrossFlow_3D import (
    NonPeriodicStructuredVelocityField3D,
    audit_indexed_vtk_reference,
    dimensionless_primitive_features,
)
from FMT_Utils.Task4B_ProxyLabels_3D import (
    CLASS_NAMES,
    build_four_vortex_type_labels,
    sample_scalar_on_voxel_grid,
)
from FMT_Utils.VoxelSegmentation_3D import load_vtk_segmentation


DEFAULT_CONFIG = Path("config/mainExp_Task4B_ChannelToTBL_2.3.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
    )
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _same_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _snapshot_config(spec: dict, output_dir: Path) -> None:
    path = output_dir / "config_snapshot.yaml"
    if path.exists():
        old = yaml.safe_load(path.read_text(encoding="utf-8"))
        if _same_json(old) != _same_json(spec):
            raise RuntimeError("frozen output config changed; use a new experiment version")
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def _candidate_vectors(field, seeds: np.ndarray, chunk_size: int):
    velocity = np.empty((len(seeds), 3), dtype=np.float32)
    vorticity = np.empty_like(velocity)
    omega_y_prime = np.empty(len(seeds), dtype=np.float32)
    for start in range(0, len(seeds), int(chunk_size)):
        stop = min(len(seeds), start + int(chunk_size))
        velocity[start:stop] = field.velocity(seeds[start:stop])
        vorticity[start:stop] = field.vorticity(seeds[start:stop])
        omega_y_prime[start:stop] = field.omega_y_prime(seeds[start:stop])
    return velocity, vorticity, omega_y_prime


def _id_recall(candidate: np.ndarray, ids: np.ndarray) -> dict:
    per_id = {}
    for vortex_id in np.unique(ids[ids > 0]):
        member = ids == int(vortex_id)
        per_id[str(int(vortex_id))] = float(np.mean(candidate[member]))
    values = np.asarray(list(per_id.values()), dtype=np.float64)
    return {
        "micro": float(np.mean(candidate[ids > 0])),
        "vortex_id_equal": float(np.mean(values)),
        "minimum_vortex_id": float(np.min(values)),
        "per_vortex_id": per_id,
    }


def _validate_curl_gate(audit: dict, gate: dict) -> None:
    correlations = np.asarray(audit["vorticity_component_correlation_xyz"])
    sign = np.asarray(audit["vorticity_component_sign_agreement_xyz"])
    relative_rmse = float(audit["vorticity_vector_relative_rmse"])
    failures = []
    scalar_checks = np.asarray(
        [
            *correlations.tolist(),
            *sign.tolist(),
            relative_rmse,
            float(audit["point_coordinate_max_abs_error"]),
            float(audit.get("stored_oyf_max_abs_error", np.nan)),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(scalar_checks).all():
        failures.append("curl/source audit contains non-finite statistics")
    if float(np.min(correlations)) < float(gate["minimum_component_correlation"]):
        failures.append(f"component correlation {correlations.tolist()}")
    if float(np.min(sign)) < float(gate["minimum_component_sign_agreement"]):
        failures.append(f"component sign agreement {sign.tolist()}")
    if relative_rmse > float(gate["maximum_vector_relative_rmse"]):
        failures.append(f"vector relative RMSE {relative_rmse}")
    if float(audit["point_coordinate_max_abs_error"]) > 1.0e-5:
        failures.append("PointIds do not map to the structured-grid coordinates")
    if float(audit.get("stored_oyf_max_abs_error", np.inf)) > 1.0e-7:
        failures.append("GT oyf differs from tbl.vtk at PointIds")
    if failures:
        raise RuntimeError("TBL curl/source audit failed: " + "; ".join(failures))


def prepare(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_config(spec, output_dir)
    cache_dir = output_dir / "target_cache"
    chunk_dir = cache_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    source_cache = Path(spec["source"]["cache"])
    source_hash = _sha256(source_cache)
    if source_hash != str(spec["source"]["cache_sha256"]):
        raise RuntimeError("channel source cache SHA-256 changed")
    flow_path = Path(spec["target"]["flow_path"])
    gt_path = Path(spec["target"]["gt_path"])
    validation_path = Path(spec["target"]["vorticity_validation_path"])
    for path in (flow_path, gt_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    voxel_spec = spec["voxelization"]
    segmentation, gt_metadata, selected_array = load_vtk_segmentation(
        gt_path,
        resolution_xyz=voxel_spec.get("resolution_xyz"),
        array_name=spec["target"]["label_array"],
        association=spec["target"]["label_association"],
        max_voxels=int(voxel_spec["preview_max_voxels"]),
        max_axis_resolution=int(voxel_spec["max_axis_resolution"]),
    )
    vortex_ids_zyx = np.asarray(segmentation.labels_zyx, dtype=np.int32)
    field, flow_metadata = NonPeriodicStructuredVelocityField3D.from_vtk(
        flow_path,
        velocity_array_name=spec["target"]["velocity_array"],
        omega_y_prime_array_name=spec["target"]["omega_y_prime_array"],
        minimum_oyf_correlation=float(spec["target"]["minimum_derived_oyf_correlation"]),
    )
    curl_audit = audit_indexed_vtk_reference(
        validation_path, field, require_vorticity=True
    )
    _validate_curl_gate(curl_audit, spec["target"]["curl_validation_gate"])

    deviation_flow = field.wall_normal_vorticity_deviation()
    sampled_deviation = sample_scalar_on_voxel_grid(
        field.axes_zyx,
        deviation_flow,
        segmentation.grid,
        chunk_size=262_144,
    )
    finite = np.isfinite(sampled_deviation)
    percentile = float(spec["target_proxy"]["percentile"])
    threshold = float(np.percentile(sampled_deviation[finite], percentile))
    candidate_before = finite & (sampled_deviation >= threshold)
    hairpin = vortex_ids_zyx > 0
    candidate_after = candidate_before | hairpin
    recall = _id_recall(candidate_before, vortex_ids_zyx)
    absolute_threshold = float(
        spec["target_proxy"]["absolute_channel_threshold_diagnostic"]
    )
    absolute_candidate = finite & (sampled_deviation >= absolute_threshold)
    absolute_recall = _id_recall(absolute_candidate, vortex_ids_zyx)

    iz, iy, ix = np.nonzero(candidate_after)
    indices_xyz = np.column_stack((ix, iy, iz)).astype(np.int32, copy=False)
    seeds = segmentation.grid.domain_min_xyz + (
        indices_xyz.astype(np.float64) + 0.5
    ) * segmentation.grid.voxel_size_xyz
    velocity, vorticity, omega_y_prime = _candidate_vectors(
        field, seeds, int(spec["streamlines"]["feature_chunk_size"])
    )
    candidate_ids = vortex_ids_zyx[iz, iy, ix]
    assigned = build_four_vortex_type_labels(
        velocity,
        vorticity,
        omega_y_prime,
        candidate_ids,
        streamwise_max_angle_degrees=float(
            spec["target_proxy"]["streamwise_max_angle_degrees"]
        ),
        minimum_vector_norm=float(spec["target_proxy"]["minimum_vector_norm"]),
    )
    labels = np.asarray(assigned["labels"], dtype=np.int8)
    label_valid = labels >= 0
    if not np.all(np.isfinite(seeds[label_valid])):
        raise RuntimeError("candidate seed coordinates contain non-finite values")
    candidate_rows = np.flatnonzero(label_valid)

    streamline_spec = spec["streamlines"]
    spatial_step = float(
        float(streamline_spec["spatial_step_mean_voxel_scale"])
        * np.mean(segmentation.grid.voxel_size_xyz)
    )
    offset = float(
        float(streamline_spec["source_offset_over_spatial_step"]) * spatial_step
    )
    encoder = spec["encoder"]
    chunk_size = int(streamline_spec["integration_chunk_size"])
    chunk_entries = []
    valid_class_counts = np.zeros(4, dtype=np.int64)
    invalid_class_counts = np.zeros(4, dtype=np.int64)
    valid_id_counts: dict[str, int] = {}
    initial_id_counts: dict[str, int] = {}
    valid_id_class_counts: dict[str, list[int]] = {}
    initial_id_class_counts: dict[str, list[int]] = {}
    valid_total = 0
    for vortex_id in np.unique(candidate_ids[candidate_rows]):
        if vortex_id > 0:
            key = str(int(vortex_id))
            initial_id_counts[key] = int(
                np.count_nonzero(candidate_ids[candidate_rows] == vortex_id)
            )
            initial_id_class_counts[key] = [
                int(
                    np.count_nonzero(
                        (candidate_ids[candidate_rows] == vortex_id)
                        & (labels[candidate_rows] == class_id)
                    )
                )
                for class_id in range(4)
            ]

    for chunk_number, start in enumerate(range(0, len(candidate_rows), chunk_size)):
        stop = min(len(candidate_rows), start + chunk_size)
        rows = candidate_rows[start:stop]
        primitives, valid = integrate_bidirectional_cross_primitives(
            field,
            seeds[rows],
            spatial_step=spatial_step,
            steps_per_direction=int(streamline_spec["steps_per_direction"]),
            offset=offset,
            minimum_speed=float(streamline_spec["minimum_speed"]),
            chunk_size=chunk_size,
            unit_velocity=True,
        )
        for class_id in range(4):
            invalid_class_counts[class_id] += int(
                np.count_nonzero((labels[rows] == class_id) & ~valid)
            )
        kept_rows = rows[valid]
        kept_primitives = primitives[valid]
        if len(kept_rows) == 0:
            print(f"chunk {chunk_number:04d}: 0/{len(rows)} valid", flush=True)
            continue
        raw, fmt = dimensionless_primitive_features(
            kept_primitives,
            spatial_step,
            num_freq=int(encoder["num_freq"]),
            neighbor_scale=float(encoder["neighbor_scale"]),
            neighbor_pool=str(encoder["neighbor_pool"]),
            mode=str(encoder["mode"]),
            include_chirality=bool(encoder["include_chirality"]),
        )
        kept_labels = labels[kept_rows]
        kept_ids = candidate_ids[kept_rows]
        valid_class_counts += np.bincount(kept_labels, minlength=4)
        for vortex_id in np.unique(kept_ids[kept_ids > 0]):
            key = str(int(vortex_id))
            valid_id_counts[key] = valid_id_counts.get(key, 0) + int(
                np.count_nonzero(kept_ids == vortex_id)
            )
            counts = valid_id_class_counts.setdefault(key, [0, 0, 0, 0])
            for class_id in range(4):
                counts[class_id] += int(
                    np.count_nonzero(
                        (kept_ids == vortex_id) & (kept_labels == class_id)
                    )
                )
        chunk_path = chunk_dir / f"target_chunk_{chunk_number:04d}.npz"
        _atomic_savez(
            chunk_path,
            raw_features=raw,
            fmt_features=fmt,
            labels=kept_labels,
            seeds_xyz=seeds[kept_rows].astype(np.float32),
            vortex_ids=kept_ids.astype(np.int32),
            voxel_indices_xyz=indices_xyz[kept_rows],
            source_candidate_indices=kept_rows.astype(np.int64),
        )
        entry = {
            "path_relative_to_manifest": str(
                chunk_path.relative_to(cache_dir).as_posix()
            ),
            "sha256": _sha256(chunk_path),
            "row_count": int(len(kept_rows)),
            "source_candidate_index_min": int(np.min(kept_rows)),
            "source_candidate_index_max": int(np.max(kept_rows)),
        }
        chunk_entries.append(entry)
        valid_total += len(kept_rows)
        print(
            f"chunk {chunk_number:04d}: {len(kept_rows)}/{len(rows)} valid; "
            f"cumulative={valid_total}",
            flush=True,
        )

    if not chunk_entries or np.any(valid_class_counts == 0):
        raise RuntimeError(
            f"TBL valid cache does not contain all four classes: {valid_class_counts}"
        )
    id_validity = {
        key: {
            "before": int(initial_id_counts[key]),
            "valid": int(valid_id_counts.get(key, 0)),
            "fraction": float(valid_id_counts.get(key, 0) / initial_id_counts[key]),
            "class_counts_before": {
                name: int(initial_id_class_counts[key][class_id])
                for class_id, name in enumerate(CLASS_NAMES)
            },
            "class_counts_valid": {
                name: int(valid_id_class_counts.get(key, [0, 0, 0, 0])[class_id])
                for class_id, name in enumerate(CLASS_NAMES)
            },
        }
        for key in sorted(initial_id_counts, key=int)
    }
    manifest = {
        "experiment": spec["experiment"],
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "source_cache": str(source_cache.resolve()),
        "source_cache_sha256": source_hash,
        "target_flow": str(flow_path.resolve()),
        "target_flow_sha256": _sha256(flow_path),
        "target_gt": str(gt_path.resolve()),
        "target_gt_sha256": _sha256(gt_path),
        "target_vorticity_validation": str(validation_path.resolve()),
        "target_vorticity_validation_sha256": _sha256(validation_path),
        "target_grid": {
            "resolution_xyz": list(segmentation.grid.resolution_xyz),
            "domain_min_xyz": segmentation.grid.domain_min_xyz.tolist(),
            "domain_max_xyz": segmentation.grid.domain_max_xyz.tolist(),
            "voxel_size_xyz": segmentation.grid.voxel_size_xyz.tolist(),
        },
        "selected_gt_array": selected_array,
        "gt_metadata": gt_metadata,
        "flow_metadata": flow_metadata,
        "curl_validation_audit": curl_audit,
        "proxy": {
            "quantity": "wall-normal-plane vorticity deviation; y deviation from stored oyf",
            "percentile": percentile,
            "threshold": threshold,
            "candidate_count_before_hairpin_override": int(np.count_nonzero(candidate_before)),
            "candidate_count_after_hairpin_override": int(np.count_nonzero(candidate_after)),
            "hairpin_voxel_count": int(np.count_nonzero(hairpin)),
            "hairpin_recall_before_override": recall,
            "absolute_channel_threshold_diagnostic": {
                "threshold": absolute_threshold,
                "candidate_count": int(np.count_nonzero(absolute_candidate)),
                "hairpin_recall": absolute_recall,
            },
            "invalid_orientation_count": int(np.count_nonzero(~label_valid)),
            "class_counts_before_streamline_validity": {
                name: int(np.count_nonzero(labels == class_id))
                for class_id, name in enumerate(CLASS_NAMES)
            },
        },
        "streamlines": {
            "spatial_step": spatial_step,
            "offset": offset,
            "offset_over_spatial_step": offset / spatial_step,
            "steps_per_direction": int(streamline_spec["steps_per_direction"]),
            "sampled_steps": 2 * int(streamline_spec["steps_per_direction"]) + 1,
            "candidate_count": int(len(candidate_rows)),
            "valid_count": int(valid_total),
            "invalid_count": int(len(candidate_rows) - valid_total),
            "valid_fraction": float(valid_total / len(candidate_rows)),
            "valid_class_counts": {
                name: int(valid_class_counts[class_id])
                for class_id, name in enumerate(CLASS_NAMES)
            },
            "invalid_class_counts": {
                name: int(invalid_class_counts[class_id])
                for class_id, name in enumerate(CLASS_NAMES)
            },
            "per_vortex_id_validity": id_validity,
            "vortex_ids_with_zero_valid_rows": [
                int(key) for key, value in id_validity.items() if value["valid"] == 0
            ],
        },
        "representation": spec["representation"],
        "encoder": spec["encoder"],
        "chunks": chunk_entries,
        "chunk_count": len(chunk_entries),
        "test_selection": "all proxy candidate cubes, followed only by common nonperiodic streamline validity",
        "target_use_boundary": "target labels/features never select normalization, model weights, epoch, or hyperparameters",
    }
    manifest_path = cache_dir / "target_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "experiment": spec["experiment"],
        "target_cache_manifest": str(manifest_path.resolve()),
        "target_cache_manifest_sha256": _sha256(manifest_path),
        "target_class_support": manifest["streamlines"]["valid_class_counts"],
        "streamline_valid_fraction": manifest["streamlines"]["valid_fraction"],
        "vortex_ids_with_zero_valid_rows": manifest["streamlines"][
            "vortex_ids_with_zero_valid_rows"
        ],
        "curl_validation_audit": curl_audit,
        "proxy_threshold": threshold,
        "proxy_hairpin_recall_before_override": recall,
    }
    (output_dir / "target_preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


if __name__ == "__main__":
    prepare(_parse_args().config)
