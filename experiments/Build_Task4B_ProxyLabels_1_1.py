"""Build frozen four-vortex-type proxy labels for channel-flow Task4-b."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from FMT_Utils.Task4A_StreamlineClustering_3D import ChannelVelocityField3D
from FMT_Utils.Task4B_ProxyLabels_3D import (
    CLASS_NAMES,
    IGNORE_LABEL,
    build_four_vortex_type_labels,
    hairpin_physics_quality_rows,
    pad_periodic_xy,
    sample_scalar_on_voxel_grid,
    scan_ivd_thresholds,
    select_recall_constrained_ivd,
    split_vortex_ids,
    split_vortex_ids_by_buffered_x,
    summarize_hairpin_physics,
    vorticity_deviation_volume,
)
from FMT_Utils.VoxelSegmentation_3D import load_vtk_segmentation


DEFAULT_CONFIG = Path("config/Verify_Task4B_ProxyLabels_1.2.yaml")


def _resolve_path(candidates: list[str], override: str | None, name: str) -> Path:
    if override is not None:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    for value in candidates:
        path = Path(value)
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"no {name} found among {candidates}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _config_identity(spec: dict) -> dict:
    return json.loads(json.dumps(spec, sort_keys=True))


def _validate_or_write_snapshot(spec: dict, output_dir: Path) -> None:
    snapshot = output_dir / "config_snapshot.yaml"
    if snapshot.exists():
        old = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        if _config_identity(old) != _config_identity(spec):
            raise RuntimeError(
                f"configuration changed for frozen output {output_dir}; "
                "use a new experiment version"
            )
    snapshot.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def _candidate_vectors(
    field: ChannelVelocityField3D,
    points_xyz: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.empty((len(points_xyz), 3), dtype=np.float32)
    vorticity = np.empty_like(velocity)
    for start in range(0, len(points_xyz), int(chunk_size)):
        stop = min(len(points_xyz), start + int(chunk_size))
        velocity[start:stop] = field.velocity(points_xyz[start:stop])
        vorticity[start:stop] = field.vorticity(points_xyz[start:stop])
    return velocity, vorticity


def _split_recall_rows(
    candidate_before_override: np.ndarray,
    vortex_ids_zyx: np.ndarray,
    id_splits: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for split, split_ids in id_splits.items():
        recalls = []
        voxel_count = 0
        retained = 0
        for vortex_id in split_ids:
            member = vortex_ids_zyx == int(vortex_id)
            count = int(np.count_nonzero(member))
            keep = int(np.count_nonzero(candidate_before_override & member))
            voxel_count += count
            retained += keep
            recalls.append(keep / count)
        rows.append(
            {
                "split": split,
                "vortex_id_count": int(len(split_ids)),
                "hairpin_voxel_count": voxel_count,
                "retained_before_override": retained,
                "micro_recall_before_override": float(retained / voxel_count),
                "vortex_equal_recall_before_override": float(np.mean(recalls)),
                "minimum_vortex_recall_before_override": float(np.min(recalls)),
                "vortex_ids_recall_ge_0_90": int(
                    np.count_nonzero(np.asarray(recalls) >= 0.90)
                ),
            }
        )
    return rows


def build(
    config_path: str | Path,
    *,
    gt_override: str | None = None,
    flow_override: str | None = None,
    output_override: str | None = None,
) -> Path:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(output_override or spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_or_write_snapshot(spec, output_dir)

    input_spec = spec["input"]
    gt_path = _resolve_path(
        input_spec["gt_path_candidates"], gt_override, "channel hairpin GT"
    )
    flow_path = _resolve_path(
        input_spec["flow_path_candidates"], flow_override, "channel flow"
    )
    voxel_spec = spec["voxelization"]
    segmentation, gt_metadata, selected_array = load_vtk_segmentation(
        gt_path,
        resolution_xyz=voxel_spec.get("resolution_xyz"),
        array_name=input_spec["label_array"],
        association=input_spec["association"],
        max_voxels=int(voxel_spec["preview_max_voxels"]),
        max_axis_resolution=int(voxel_spec["max_axis_resolution"]),
    )
    vortex_ids_zyx = segmentation.labels_zyx

    field, flow_metadata = ChannelVelocityField3D.from_vtk(flow_path)
    split_spec = spec["splits"]
    if split_spec.get("unit") == "complete_VortexId_inside_buffered_x_interval":
        nx_target = segmentation.grid.resolution_xyz[0]
        x_centers = segmentation.grid.domain_min_xyz[0] + (
            np.arange(nx_target, dtype=np.float64) + 0.5
        ) * segmentation.grid.voxel_size_xyz[0]
        id_splits, excluded_vortex_ids = split_vortex_ids_by_buffered_x(
            vortex_ids_zyx,
            x_centers,
            x_intervals=split_spec["x_intervals"],
            period_x=field.periods_xy[0],
            domain_min_x=segmentation.grid.domain_min_xyz[0],
        )
    else:
        id_splits = split_vortex_ids(
            vortex_ids_zyx,
            train_fraction=float(split_spec["train_fraction"]),
            validation_fraction=float(split_spec["validation_fraction"]),
            seed=int(split_spec["seed"]),
        )
        excluded_vortex_ids = np.empty(0, dtype=np.int32)

    nx, ny, nz = (int(value) for value in flow_metadata["dimensions_xyz"])
    raw_vorticity = np.asarray(
        field.vorticity_zyx3[:nz, :ny, :nx], dtype=np.float32
    )
    search_spec = spec.get("vorticity_deviation_search", spec.get("ivd_search"))
    if search_spec is None:
        raise KeyError("missing vorticity_deviation_search")
    start, stop, step = (
        float(value) for value in search_spec["percentile_grid"]
    )
    percentiles = np.arange(start, stop + 0.5 * step, step, dtype=np.float64)
    sampled_by_mode: dict[str, np.ndarray] = {}
    rows_by_mode: dict[str, list[dict]] = {}
    mode_means: dict[str, np.ndarray] = {}
    for mode in search_spec["modes"]:
        deviation, mean_field = vorticity_deviation_volume(raw_vorticity, mode)
        sampled = sample_scalar_on_voxel_grid(
            field.axes_zyx,
            pad_periodic_xy(deviation),
            segmentation.grid,
            chunk_size=int(search_spec["interpolation_chunk_size"]),
        )
        sampled_by_mode[str(mode)] = sampled
        mode_means[str(mode)] = mean_field
        reference = (
            sampled[np.isfinite(sampled)]
            if search_spec["threshold_reference"] == "target_voxel_grid"
            else deviation.reshape(-1)
        )
        rows_by_mode[str(mode)] = scan_ivd_thresholds(
            reference,
            sampled,
            vortex_ids_zyx,
            id_splits["train"],
            percentiles,
        )
    selection, threshold_rows = select_recall_constrained_ivd(
        rows_by_mode,
        minimum_train_hairpin_recall=float(
            search_spec["minimum_train_micro_recall"]
        ),
        minimum_train_vortex_equal_recall=float(
            search_spec["minimum_train_vortex_equal_recall"]
        ),
    )
    _write_csv(
        output_dir / "vorticity_deviation_threshold_search.csv", threshold_rows
    )

    selected_ivd = sampled_by_mode[selection.mode]
    candidate_before = np.isfinite(selected_ivd) & (
        selected_ivd >= selection.threshold
    )
    hairpin = vortex_ids_zyx > 0
    candidate_after = candidate_before | hairpin
    split_recall_rows = _split_recall_rows(
        candidate_before, vortex_ids_zyx, id_splits
    )
    _write_csv(
        output_dir / "vorticity_deviation_split_recall.csv", split_recall_rows
    )

    iz, iy, ix = np.nonzero(candidate_after)
    candidate_indices_xyz = np.column_stack((ix, iy, iz)).astype(
        np.int32, copy=False
    )
    candidate_seeds_xyz = segmentation.grid.domain_min_xyz + (
        candidate_indices_xyz + 0.5
    ) * segmentation.grid.voxel_size_xyz
    candidate_velocity, candidate_vorticity = _candidate_vectors(
        field,
        candidate_seeds_xyz,
        chunk_size=int(search_spec["interpolation_chunk_size"]),
    )
    plane_omega_y = raw_vorticity[..., 1].mean(axis=(1, 2))
    omega_y_prime = candidate_vorticity[:, 1] - np.interp(
        candidate_seeds_xyz[:, 2], field.axes_zyx[0], plane_omega_y
    )

    proxy_spec = spec["proxy_labels"]
    assigned = build_four_vortex_type_labels(
        candidate_velocity,
        candidate_vorticity,
        omega_y_prime,
        vortex_ids_zyx[iz, iy, ix],
        streamwise_max_angle_degrees=float(
            proxy_spec["streamwise_max_angle_degrees"]
        ),
        minimum_vector_norm=float(proxy_spec["minimum_vector_norm"]),
    )
    candidate_labels = assigned["labels"]
    proxy_labels_zyx = np.full(
        segmentation.grid.shape_zyx, IGNORE_LABEL, dtype=np.int8
    )
    proxy_labels_zyx[iz, iy, ix] = candidate_labels

    candidate_vortex_ids = vortex_ids_zyx[iz, iy, ix]
    annotated = candidate_vortex_ids > 0
    physics_rows = hairpin_physics_quality_rows(
        candidate_seeds_xyz[annotated],
        candidate_vorticity[annotated],
        omega_y_prime[annotated],
        candidate_vortex_ids[annotated],
        candidate_labels[annotated],
        periods_xy=field.periods_xy,
    )
    _write_csv(output_dir / "hairpin_physics_quality.csv", physics_rows)
    physics_summary = summarize_hairpin_physics(physics_rows)

    result_path = output_dir / "task4b_proxy_labels.npz"
    np.savez_compressed(
        result_path,
        resolution_xyz=np.asarray(segmentation.grid.resolution_xyz, dtype=np.int32),
        domain_min_xyz=segmentation.grid.domain_min_xyz,
        domain_max_xyz=segmentation.grid.domain_max_xyz,
        vortex_ids_zyx=vortex_ids_zyx,
        selected_vorticity_deviation_zyx=selected_ivd,
        vorticity_deviation_mode=np.asarray(selection.mode),
        vorticity_deviation_status=np.asarray(
            "standard_IVD"
            if selection.mode == "whole_domain"
            else "channel_profile_vorticity_deviation_not_standard_IVD"
        ),
        vorticity_deviation_threshold=np.asarray(selection.threshold),
        vorticity_deviation_percentile=np.asarray(selection.percentile),
        candidate_before_hairpin_override_zyx=candidate_before,
        candidate_after_hairpin_override_zyx=candidate_after,
        proxy_labels_zyx=proxy_labels_zyx,
        candidate_indices_xyz=candidate_indices_xyz,
        candidate_seeds_xyz=candidate_seeds_xyz.astype(np.float32),
        candidate_vortex_ids=candidate_vortex_ids,
        candidate_labels=candidate_labels,
        candidate_vorticity_deviation=selected_ivd[iz, iy, ix],
        candidate_velocity_xyz=candidate_velocity,
        candidate_vorticity_xyz=candidate_vorticity,
        candidate_omega_y_prime=omega_y_prime.astype(np.float32),
        candidate_orientation_angle_degrees=assigned[
            "orientation_angle_degrees"
        ],
        candidate_valid_orientation=assigned["valid_orientation"],
        train_vortex_ids=id_splits["train"],
        validation_vortex_ids=id_splits["validation"],
        test_vortex_ids=id_splits["test"],
        excluded_buffer_vortex_ids=excluded_vortex_ids,
    )

    class_counts = {
        name: int(np.count_nonzero(candidate_labels == class_id))
        for class_id, name in enumerate(CLASS_NAMES)
    }
    summary = {
        "experiment": spec["experiment"],
        "task": spec["task"],
        "config": str(config_path.resolve()),
        "result_path": str(result_path.resolve()),
        "taxonomy": {
            "protocol": "vortex_only_four_types",
            "ignore_value": IGNORE_LABEL,
            "classes": {str(index): name for index, name in enumerate(CLASS_NAMES)},
            "non_vortex_policy": (
                "cubes below the selected vorticity-deviation gate are excluded, "
                "not a fifth class"
            ),
        },
        "gt": gt_metadata,
        "selected_gt_array": selected_array,
        "flow": flow_metadata,
        "vortex_id_splits": {
            key: [int(value) for value in values]
            for key, values in id_splits.items()
        },
        "excluded_buffer_vortex_ids": [
            int(value) for value in excluded_vortex_ids
        ],
        "vorticity_deviation_selection": {
            "selected_mode": selection.mode,
            "selected_mode_status": (
                "standard_IVD"
                if selection.mode == "whole_domain"
                else "channel_profile_vorticity_deviation_not_standard_IVD"
            ),
            "threshold_reference": search_spec["threshold_reference"],
            "percentile": selection.percentile,
            "threshold": selection.threshold,
            "train_micro_recall": selection.train_hairpin_recall,
            "train_vortex_equal_recall": selection.train_vortex_equal_recall,
            "candidate_fraction_before_override": selection.candidate_fraction,
            "candidate_count_before_override": int(np.count_nonzero(candidate_before)),
            "candidate_count_after_override": int(np.count_nonzero(candidate_after)),
            "hairpin_override_count": int(
                np.count_nonzero(hairpin & ~candidate_before)
            ),
            "split_recall_before_override": split_recall_rows,
            "selection_warning": (
                "Outside the manual hairpin annotation contains unknown ordinary "
                "vortices, so candidate compactness is not precision."
            ),
        },
        "proxy_rules": {
            "ordinary": "velocity-curl absolute angle <=45 streamwise, >45 spanwise",
            "head": (
                "VortexId>0 AND angle>45 AND omega_y_prime>0 AND "
                "abs(omega_y_prime)>=max(abs(omega_x),abs(omega_z))"
            ),
            "limb": "all remaining VortexId>0 voxels; includes legs, necks, transitions",
            "omega_y_prime": "omega_y minus x-y plane mean omega_y at the same z",
        },
        "counts": {
            "grid_voxels": int(vortex_ids_zyx.size),
            "manual_hairpin_voxels": int(np.count_nonzero(hairpin)),
            "valid_four_class_voxels": int(
                np.count_nonzero(candidate_labels != IGNORE_LABEL)
            ),
            "invalid_orientation_ignored": int(
                np.count_nonzero(candidate_labels == IGNORE_LABEL)
            ),
            "per_class": class_counts,
        },
        "hairpin_physics_quality": physics_summary,
        "interpretation_boundary": (
            "These labels merge a deterministic vorticity-deviation gate, velocity-curl, "
            "and manual hairpin support. They measure proxy reproduction and are not "
            "manual head/limb ground truth."
        ),
    }
    whole_domain_reference = vorticity_deviation_volume(
        raw_vorticity, "whole_domain"
    )[0]
    flow_grid_p95 = float(np.percentile(whole_domain_reference, 95.0))
    whole_domain_sampled = sampled_by_mode["whole_domain"]
    summary["standard_whole_domain_ivd_p95_diagnostic"] = {
        "threshold_reference": "flow_grid",
        "threshold": flow_grid_p95,
        "annotated_hairpin_recall": float(
            np.mean(whole_domain_sampled[hairpin] >= flow_grid_p95)
        ),
        "candidate_fraction_target_grid": float(
            np.mean(whole_domain_sampled >= flow_grid_p95)
        ),
        "status": "diagnostic_only_not_selected_for_Task4B_proxy",
    }
    manifest = {
        "result": str(result_path.resolve()),
        "selected_quantity": summary["vorticity_deviation_selection"],
        "array_definitions": {
            "selected_vorticity_deviation_zyx": (
                "selected whole-domain IVD or channel-profile vorticity "
                "deviation, disambiguated by vorticity_deviation_status"
            ),
            "candidate_vorticity_deviation": (
                "selected_vorticity_deviation_zyx at candidate_indices_xyz"
            ),
        },
    }
    (output_dir / "proxy_label_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(summary["vorticity_deviation_selection"], indent=2),
        flush=True,
    )
    print(json.dumps(summary["counts"], indent=2), flush=True)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--gt")
    parser.add_argument("--flow")
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build(
        args.config,
        gt_override=args.gt,
        flow_override=args.flow,
        output_override=args.output_dir,
    )
