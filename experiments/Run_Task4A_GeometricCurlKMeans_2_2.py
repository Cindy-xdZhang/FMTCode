"""Tune KMeans on velocity--curl orientation reconstructed from streamline geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from FMT_Utils.Task4A_PerVortexClustering_3D import (
    map_clusters_by_geometry_score,
    map_clusters_by_proxy_permutation,
    per_vortex_kmeans,
    velocity_curl_from_time_parameterized_cross,
)
from FMT_Utils.Task4A_StreamlineClustering_3D import (
    ChannelVelocityField3D,
    integrate_bidirectional_cross_primitives,
)
from FMT_Utils.Task4A_TraditionalBaseline_3D import proxy_groundtruth_agreement


DEFAULT_CONFIG = Path("config/Verify_Task4A_GeometricCurlKMeans_2.2.yaml")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_flow(candidates: list[str], override: str | None) -> Path:
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    path = next((Path(value) for value in candidates if Path(value).is_file()), None)
    if path is None:
        raise FileNotFoundError(f"no channel velocity field found among {candidates}")
    return path.resolve()


def _metric_means(summary: dict) -> tuple[float, float, float]:
    return (
        float(summary["primary_vortex_equal_weighted_macro_f1"]["mean"]),
        float(summary["vortex_equal_weighted_hard_label_mse"]["mean"]),
        float(
            summary[
                "secondary_vortex_equal_weighted_angle_confidence_weighted_mse"
            ]["mean"]
        ),
    )


def _integrate_and_reconstruct(
    field: ChannelVelocityField3D,
    seeds_xyz: np.ndarray,
    *,
    parameter_step: float,
    steps_per_direction: int,
    cross_offset: float,
    minimum_speed: float,
    minimum_vector_norm: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    primitives, valid_primitive = integrate_bidirectional_cross_primitives(
        field,
        seeds_xyz,
        spatial_step=float(parameter_step),
        steps_per_direction=int(steps_per_direction),
        offset=float(cross_offset),
        minimum_speed=float(minimum_speed),
        chunk_size=int(chunk_size),
        unit_velocity=False,
    )
    reconstructed = velocity_curl_from_time_parameterized_cross(
        primitives,
        parameter_step=float(parameter_step),
        cross_offset=float(cross_offset),
        minimum_vector_norm=float(minimum_vector_norm),
    )
    valid = valid_primitive & reconstructed["valid"]
    if not valid.all():
        raise RuntimeError(
            f"time-parameterized geometry left {int(np.count_nonzero(~valid))} invalid cubes"
        )
    return primitives, valid, reconstructed


def _evaluate_kmeans(
    geometry_score: np.ndarray,
    vortex_ids: np.ndarray,
    proxy_classes: np.ndarray,
    proxy_angles: np.ndarray,
    *,
    temperature: float,
    kmeans_config: dict,
    seed: int,
) -> tuple[np.ndarray, dict, dict, list[dict]]:
    signed_margin = geometry_score - 0.5
    embedding = np.tanh(signed_margin / float(temperature))[:, None]
    raw_clusters, diagnostics = per_vortex_kmeans(
        embedding,
        vortex_ids,
        base_feature_width=1,
        neighbor_weight=1.0,
        seed=int(seed),
        **kmeans_config,
    )
    semantic, mapping = map_clusters_by_geometry_score(
        raw_clusters, vortex_ids, geometry_score
    )
    _, geometry_summary = proxy_groundtruth_agreement(
        semantic, proxy_classes, vortex_ids, proxy_angles
    )
    geometry_summary["semantic_mapping"] = (
        "Per-VortexId KMeans raw clusters are named by the higher median "
        "reconstructed cos^2(angle) streamline-geometry score; direct velocity, "
        "curl, and proxy labels are not used for naming."
    )
    oracle_semantic, _ = map_clusters_by_proxy_permutation(
        raw_clusters, proxy_classes, vortex_ids
    )
    _, oracle_summary = proxy_groundtruth_agreement(
        oracle_semantic, proxy_classes, vortex_ids, proxy_angles
    )
    oracle_summary["semantic_mapping"] = (
        "Oracle external evaluation chooses the better proxy label permutation "
        "inside each VortexId; this is not a deployable naming rule."
    )
    return semantic, geometry_summary, oracle_summary, [
        {**row, **mapping[index]} for index, row in enumerate(diagnostics)
    ]


def _candidate_row(
    *,
    phase: str,
    time_scale: float,
    offset_scale: float,
    steps: int,
    temperature: float,
    kmeans_id: str,
    kmeans_config: dict,
    reconstruction_summary: dict,
    geometry_summary: dict,
    oracle_summary: dict,
    angle_mae: float,
) -> dict:
    reconstruction_f1, reconstruction_mse, _ = _metric_means(reconstruction_summary)
    geometry_f1, geometry_mse, geometry_angle_mse = _metric_means(geometry_summary)
    oracle_f1, oracle_mse, _ = _metric_means(oracle_summary)
    return {
        "phase": phase,
        "time_step_mean_voxel_over_median_speed_scale": float(time_scale),
        "cross_offset_min_voxel_scale": float(offset_scale),
        "steps_per_direction": int(steps),
        "signed_margin_temperature": float(temperature),
        "kmeans_id": str(kmeans_id),
        "kmeans_init": str(kmeans_config["init"]),
        "kmeans_n_init": int(kmeans_config["n_init"]),
        "kmeans_max_iter": int(kmeans_config["max_iter"]),
        "kmeans_tol": float(kmeans_config["tol"]),
        "kmeans_algorithm": str(kmeans_config["algorithm"]),
        "reconstructed_angle_mae_degrees": float(angle_mae),
        "reconstruction_threshold_macro_f1": reconstruction_f1,
        "reconstruction_threshold_hard_label_mse": reconstruction_mse,
        "geometry_named_kmeans_macro_f1": geometry_f1,
        "geometry_named_kmeans_hard_label_mse": geometry_mse,
        "geometry_named_kmeans_angle_weighted_mse": geometry_angle_mse,
        "proxy_permutation_kmeans_macro_f1": oracle_f1,
        "proxy_permutation_kmeans_hard_label_mse": oracle_mse,
    }


def _selection_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row["geometry_named_kmeans_macro_f1"]),
        -float(row["geometry_named_kmeans_hard_label_mse"]),
        -float(row["reconstructed_angle_mae_degrees"]),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--flow")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> dict:
    args = _parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(config["input"]["fmt_result"])
    proxy_path = Path(config["input"]["proxy_result"])
    source = np.load(source_path)
    proxy = np.load(proxy_path)
    indices_xyz = np.asarray(source["voxel_indices_xyz"], dtype=np.int32)
    seeds_xyz = np.asarray(source["seeds_xyz"], dtype=np.float64)
    vortex_ids = np.asarray(source["vortex_ids"], dtype=np.int32)
    proxy_classes = np.asarray(proxy["baseline_classes"], dtype=np.int8)
    proxy_angles = np.asarray(proxy["absolute_velocity_curl_angle_degrees"], dtype=float)
    if not np.array_equal(indices_xyz, proxy["voxel_indices_xyz"]):
        raise RuntimeError("FMT and proxy voxel indices differ")

    resolution_xyz = np.asarray(source["resolution_xyz"], dtype=np.int32)
    domain_min_xyz = np.asarray(source["domain_min_xyz"], dtype=float)
    domain_max_xyz = np.asarray(source["domain_max_xyz"], dtype=float)
    voxel_size_xyz = (domain_max_xyz - domain_min_xyz) / resolution_xyz
    mean_voxel_size = float(np.mean(voxel_size_xyz))
    minimum_voxel_size = float(np.min(voxel_size_xyz))
    median_seed_speed = float(
        np.median(np.linalg.norm(np.asarray(proxy["velocity_xyz"]), axis=1))
    )
    reference_time_step = mean_voxel_size / median_seed_speed
    flow_path = _resolve_flow(config["input"]["flow_path_candidates"], args.flow)
    field, flow_metadata = ChannelVelocityField3D.from_vtk(flow_path)

    search = config["geometry_search"]
    phase1_kmeans = config["phase1_kmeans"]
    rows: list[dict] = []
    cache: dict[tuple[float, float], dict] = {}
    invalid_candidates: list[dict] = []
    phase1_steps = int(search["phase1_steps_per_direction"])
    for time_scale in search["time_step_mean_voxel_over_median_speed_scales"]:
        for offset_scale in search["cross_offset_min_voxel_scales"]:
            time_scale = float(time_scale)
            offset_scale = float(offset_scale)
            try:
                primitives, valid, reconstructed = _integrate_and_reconstruct(
                    field,
                    seeds_xyz,
                    parameter_step=time_scale * reference_time_step,
                    steps_per_direction=phase1_steps,
                    cross_offset=offset_scale * minimum_voxel_size,
                    minimum_speed=float(search["minimum_speed"]),
                    minimum_vector_norm=float(search["minimum_vector_norm"]),
                    chunk_size=int(search["chunk_size"]),
                )
            except RuntimeError as error:
                invalid_candidates.append(
                    {
                        "time_step_mean_voxel_over_median_speed_scale": time_scale,
                        "cross_offset_min_voxel_scale": offset_scale,
                        "steps_per_direction": phase1_steps,
                        "reason": str(error),
                    }
                )
                print(json.dumps({
                    "invalid_geometry": [time_scale, offset_scale],
                    "reason": str(error),
                }), flush=True)
                continue
            geometry_score = np.square(reconstructed["absolute_cosine"].astype(float))
            reconstructed_classes = np.where(geometry_score >= 0.5, 1, 2).astype(np.int8)
            _, reconstruction_summary = proxy_groundtruth_agreement(
                reconstructed_classes, proxy_classes, vortex_ids, proxy_angles
            )
            reconstruction_summary["semantic_mapping"] = (
                "Classes use the frozen 45-degree boundary on the orientation "
                "angle reconstructed only from streamline-cross coordinates."
            )
            angle_mae = float(np.mean(np.abs(
                reconstructed["angle_degrees"].astype(float) - proxy_angles
            )))
            cache[(time_scale, offset_scale)] = {
                "geometry_score": geometry_score,
                "reconstructed": reconstructed,
            }
            for temperature in search["signed_margin_temperatures"]:
                _, geometry_summary, oracle_summary, _ = _evaluate_kmeans(
                    geometry_score,
                    vortex_ids,
                    proxy_classes,
                    proxy_angles,
                    temperature=float(temperature),
                    kmeans_config=phase1_kmeans,
                    seed=int(config["seed"]),
                )
                rows.append(
                    _candidate_row(
                        phase="phase1_geometry",
                        time_scale=time_scale,
                        offset_scale=offset_scale,
                        steps=phase1_steps,
                        temperature=float(temperature),
                        kmeans_id="phase1_fixed",
                        kmeans_config=phase1_kmeans,
                        reconstruction_summary=reconstruction_summary,
                        geometry_summary=geometry_summary,
                        oracle_summary=oracle_summary,
                        angle_mae=angle_mae,
                    )
                )
            _write_csv(output_dir / "search_progress.csv", rows)
            print(json.dumps({
                "completed_geometry": [time_scale, offset_scale],
                "candidate_count": len(rows),
                "best_geometry_named_macro_f1": max(
                    row["geometry_named_kmeans_macro_f1"] for row in rows
                ),
                "best_reconstruction_threshold_macro_f1": max(
                    row["reconstruction_threshold_macro_f1"] for row in rows
                ),
            }), flush=True)

    phase1_winner = max(rows, key=_selection_key)
    winning_time_scale = float(
        phase1_winner["time_step_mean_voxel_over_median_speed_scale"]
    )
    winning_offset_scale = float(phase1_winner["cross_offset_min_voxel_scale"])
    winning_temperature = float(phase1_winner["signed_margin_temperature"])
    final_cache = {}
    for steps in config["phase2_steps_per_direction"]:
        primitives, valid, reconstructed = _integrate_and_reconstruct(
            field,
            seeds_xyz,
            parameter_step=winning_time_scale * reference_time_step,
            steps_per_direction=int(steps),
            cross_offset=winning_offset_scale * minimum_voxel_size,
            minimum_speed=float(search["minimum_speed"]),
            minimum_vector_norm=float(search["minimum_vector_norm"]),
            chunk_size=int(search["chunk_size"]),
        )
        geometry_score = np.square(reconstructed["absolute_cosine"].astype(float))
        reconstructed_classes = np.where(geometry_score >= 0.5, 1, 2).astype(np.int8)
        reconstruction_rows, reconstruction_summary = proxy_groundtruth_agreement(
            reconstructed_classes, proxy_classes, vortex_ids, proxy_angles
        )
        reconstruction_summary["semantic_mapping"] = (
            "Classes use the frozen 45-degree boundary on the orientation angle "
            "reconstructed only from streamline-cross coordinates."
        )
        angle_mae = float(np.mean(np.abs(
            reconstructed["angle_degrees"].astype(float) - proxy_angles
        )))
        for variant_with_id in config["phase2_kmeans_variants"]:
            variant = dict(variant_with_id)
            variant_id = str(variant.pop("id"))
            semantic, geometry_summary, oracle_summary, diagnostics = _evaluate_kmeans(
                geometry_score,
                vortex_ids,
                proxy_classes,
                proxy_angles,
                temperature=winning_temperature,
                kmeans_config=variant,
                seed=int(config["seed"]),
            )
            row = _candidate_row(
                phase="phase2_length_kmeans",
                time_scale=winning_time_scale,
                offset_scale=winning_offset_scale,
                steps=int(steps),
                temperature=winning_temperature,
                kmeans_id=variant_id,
                kmeans_config=variant,
                reconstruction_summary=reconstruction_summary,
                geometry_summary=geometry_summary,
                oracle_summary=oracle_summary,
                angle_mae=angle_mae,
            )
            rows.append(row)
            final_cache[(int(steps), variant_id)] = {
                "primitives": primitives,
                "valid": valid,
                "reconstructed": reconstructed,
                "geometry_score": geometry_score,
                "reconstructed_classes": reconstructed_classes,
                "reconstruction_rows": reconstruction_rows,
                "reconstruction_summary": reconstruction_summary,
                "semantic": semantic,
                "geometry_summary": geometry_summary,
                "oracle_summary": oracle_summary,
                "diagnostics": diagnostics,
            }

    parameter_search_path = output_dir / "parameter_search.csv"
    _write_csv(parameter_search_path, rows)
    phase2_rows = [row for row in rows if row["phase"] == "phase2_length_kmeans"]
    winner = max(phase2_rows, key=_selection_key)
    winner_cache = final_cache[(int(winner["steps_per_direction"]), winner["kmeans_id"])]

    geometry_rows, final_geometry_summary = proxy_groundtruth_agreement(
        winner_cache["semantic"], proxy_classes, vortex_ids, proxy_angles
    )
    final_geometry_summary["semantic_mapping"] = (
        "Per-VortexId KMeans raw clusters are named by the higher median "
        "reconstructed cos^2(angle) streamline-geometry score; direct velocity, "
        "curl, and proxy labels are not used for naming."
    )
    _write_csv(output_dir / "per_vortex_geometry_kmeans_metrics.csv", geometry_rows)
    _write_csv(
        output_dir / "per_vortex_reconstruction_threshold_metrics.csv",
        winner_cache["reconstruction_rows"],
    )
    _write_csv(
        output_dir / "per_vortex_kmeans_diagnostics.csv", winner_cache["diagnostics"]
    )

    result_path = output_dir / "geometric_curl_kmeans_result.npz"
    np.savez_compressed(
        result_path,
        resolution_xyz=resolution_xyz,
        domain_min_xyz=domain_min_xyz,
        domain_max_xyz=domain_max_xyz,
        voxel_indices_xyz=indices_xyz,
        seeds_xyz=seeds_xyz.astype(np.float32),
        vortex_ids=vortex_ids,
        primitive_valid=winner_cache["valid"],
        center_streamlines=winner_cache["primitives"][:, 0],
        reconstructed_velocity_xyz=winner_cache["reconstructed"]["velocity_xyz"],
        reconstructed_curl_xyz=winner_cache["reconstructed"]["curl_xyz"],
        reconstructed_angle_degrees=winner_cache["reconstructed"]["angle_degrees"],
        reconstructed_streamwise_geometry_score=winner_cache["geometry_score"],
        reconstructed_threshold_classes=winner_cache["reconstructed_classes"],
        geometry_kmeans_classes=winner_cache["semantic"],
        proxy_classes=proxy_classes,
        proxy_angle_degrees=proxy_angles,
    )

    macro_f1, hard_mse, _ = _metric_means(final_geometry_summary)
    selection = config["selection"]
    summary = {
        "experiment": config["experiment"],
        "task": config["task"],
        "config": str(config_path.resolve()),
        "source_fmt_result": str(source_path.resolve()),
        "source_proxy_result": str(proxy_path.resolve()),
        "flow": flow_metadata,
        "reference_time_step": reference_time_step,
        "median_seed_speed": median_seed_speed,
        "candidate_count": len(rows),
        "invalid_geometry_candidates": invalid_candidates,
        "phase1_winner": phase1_winner,
        "winner": winner,
        "final_geometry_named_kmeans_metrics": final_geometry_summary,
        "final_reconstruction_threshold_metrics": winner_cache[
            "reconstruction_summary"
        ],
        "success_thresholds": {
            "macro_f1_at_least": float(selection["success_macro_f1_at_least"]),
            "hard_label_mse_at_most": float(
                selection["success_hard_label_mse_at_most"]
            ),
            "passed": bool(
                macro_f1 >= float(selection["success_macro_f1_at_least"])
                and hard_mse <= float(selection["success_hard_label_mse_at_most"])
            ),
        },
        "selection_status": selection["status"],
        "method_boundary": (
            "KMeans sees only a saturated signed orientation margin reconstructed "
            "from time-parameterized streamline-cross coordinates. The 45-degree "
            "boundary is inherited from the proxy, but direct velocity/curl arrays "
            "are not clustering inputs."
        ),
        "parameter_search_path": str(parameter_search_path.resolve()),
        "result_path": str(result_path.resolve()),
    }
    summary_path = output_dir / "summary.json"
    summary["summary_path"] = str(summary_path.resolve())
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
