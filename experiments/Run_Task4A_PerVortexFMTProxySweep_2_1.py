"""Tune per-VortexId FMT KMeans geometry partitions against the physical proxy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score
import torch
import yaml

from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.Task4A_PerVortexClustering_3D import (
    fmt_base_feature_width,
    map_clusters_by_hairpin_topology,
    map_clusters_by_proxy_permutation,
    map_clusters_by_vorticity_axis,
    per_vortex_kmeans,
)
from FMT_Utils.Task4A_StreamlineClustering_3D import (
    ChannelVelocityField3D,
    fill_invalid_predictions_within_vortex,
    integrate_bidirectional_cross_primitives,
)
from FMT_Utils.Task4A_TraditionalBaseline_3D import proxy_groundtruth_agreement


DEFAULT_CONFIG = Path("config/Verify_Task4A_PerVortexFMTProxySweep_2.1.yaml")


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


def _equal_weighted_ari(
    raw_clusters: np.ndarray, proxy_classes: np.ndarray, vortex_ids: np.ndarray
) -> float:
    values = []
    for vortex_id in np.unique(vortex_ids):
        member = vortex_ids == vortex_id
        values.append(adjusted_rand_score(proxy_classes[member], raw_clusters[member]))
    return float(np.mean(values))


def _evaluate_partition(
    raw_clusters_valid: np.ndarray,
    valid: np.ndarray,
    vortex_ids: np.ndarray,
    indices_xyz: np.ndarray,
    proxy_classes: np.ndarray,
    angles: np.ndarray,
    vorticity: np.ndarray,
    topology_config: dict,
) -> tuple[dict, dict]:
    raw_full = np.full(len(vortex_ids), -1, dtype=np.int8)
    raw_full[valid] = raw_clusters_valid

    semantic_valid_by_mode = {}
    mapping_rows = {}
    semantic_valid_by_mode["proxy_permutation"], mapping_rows["proxy_permutation"] = (
        map_clusters_by_proxy_permutation(
            raw_clusters_valid, proxy_classes[valid], vortex_ids[valid]
        )
    )
    semantic_valid_by_mode["vorticity_axis"], mapping_rows["vorticity_axis"] = (
        map_clusters_by_vorticity_axis(
            raw_clusters_valid, vortex_ids[valid], vorticity[valid]
        )
    )
    semantic_valid_by_mode["hairpin_topology"], mapping_rows["hairpin_topology"] = (
        map_clusters_by_hairpin_topology(
            raw_clusters_valid,
            vortex_ids[valid],
            indices_xyz[valid],
            dominant_min_voxels=int(
                topology_config["dominant_component_min_voxels"]
            ),
            dominant_min_fraction=float(
                topology_config["dominant_component_min_fraction_of_vortex"]
            ),
        )
    )

    summaries = {}
    semantic_full = {}
    per_vortex_metrics = {}
    for mode, semantic_valid in semantic_valid_by_mode.items():
        before_fill = np.zeros(len(vortex_ids), dtype=np.int8)
        before_fill[valid] = semantic_valid
        after_fill, fill_metadata = fill_invalid_predictions_within_vortex(
            before_fill, valid, vortex_ids, indices_xyz
        )
        rows, summary = proxy_groundtruth_agreement(
            after_fill, proxy_classes, vortex_ids, angles
        )
        summary["invalid_fill"] = fill_metadata
        summaries[mode] = summary
        semantic_full[mode] = after_fill
        per_vortex_metrics[mode] = rows

    return {
        "raw_clusters_full": raw_full,
        "semantic_classes": semantic_full,
        "mapping_rows": mapping_rows,
        "per_vortex_metrics": per_vortex_metrics,
    }, summaries


def _candidate_row(
    *,
    phase: str,
    step_scale: float,
    steps: int,
    neighbor_weight: float,
    kmeans_id: str,
    kmeans_config: dict,
    valid: np.ndarray,
    raw_clusters_valid: np.ndarray,
    proxy_classes: np.ndarray,
    vortex_ids: np.ndarray,
    summaries: dict,
) -> dict:
    oracle = summaries["proxy_permutation"]
    axis = summaries["vorticity_axis"]
    topology = summaries["hairpin_topology"]
    return {
        "phase": phase,
        "spatial_step_mean_voxel_scale": float(step_scale),
        "steps_per_direction": int(steps),
        "half_integral_length_mean_voxel_scale": float(step_scale * steps),
        "sample_count_per_line": int(2 * steps + 1),
        "neighbor_weight_after_standardization": float(neighbor_weight),
        "kmeans_id": str(kmeans_id),
        "kmeans_scaling": str(kmeans_config["scaling"]),
        "kmeans_init": str(kmeans_config["init"]),
        "kmeans_n_init": int(kmeans_config["n_init"]),
        "kmeans_max_iter": int(kmeans_config["max_iter"]),
        "kmeans_tol": float(kmeans_config["tol"]),
        "kmeans_algorithm": str(kmeans_config["algorithm"]),
        "valid_primitive_count": int(np.count_nonzero(valid)),
        "valid_primitive_fraction": float(np.mean(valid)),
        "proxy_permutation_macro_f1": float(
            oracle["primary_vortex_equal_weighted_macro_f1"]["mean"]
        ),
        "proxy_permutation_hard_label_mse": float(
            oracle["vortex_equal_weighted_hard_label_mse"]["mean"]
        ),
        "proxy_permutation_angle_weighted_mse": float(
            oracle[
                "secondary_vortex_equal_weighted_angle_confidence_weighted_mse"
            ]["mean"]
        ),
        "vorticity_axis_macro_f1": float(
            axis["primary_vortex_equal_weighted_macro_f1"]["mean"]
        ),
        "vorticity_axis_hard_label_mse": float(
            axis["vortex_equal_weighted_hard_label_mse"]["mean"]
        ),
        "hairpin_topology_macro_f1": float(
            topology["primary_vortex_equal_weighted_macro_f1"]["mean"]
        ),
        "hairpin_topology_hard_label_mse": float(
            topology["vortex_equal_weighted_hard_label_mse"]["mean"]
        ),
        "vortex_equal_weighted_adjusted_rand_index": _equal_weighted_ari(
            raw_clusters_valid, proxy_classes[valid], vortex_ids[valid]
        ),
    }


def _selection_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row["proxy_permutation_macro_f1"]),
        -float(row["proxy_permutation_angle_weighted_mse"]),
        float(row["vortex_equal_weighted_adjusted_rand_index"]),
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
    if not source_path.is_file() or not proxy_path.is_file():
        raise FileNotFoundError("frozen FMT and proxy results must exist")
    source = np.load(source_path)
    proxy = np.load(proxy_path)
    indices_xyz = np.asarray(source["voxel_indices_xyz"], dtype=np.int32)
    seeds_xyz = np.asarray(source["seeds_xyz"], dtype=np.float64)
    vortex_ids = np.asarray(source["vortex_ids"], dtype=np.int32)
    proxy_classes = np.asarray(proxy["baseline_classes"], dtype=np.int8)
    angles = np.asarray(proxy["absolute_velocity_curl_angle_degrees"], dtype=float)
    vorticity = np.asarray(proxy["curl_xyz"], dtype=float)
    if not np.array_equal(indices_xyz, proxy["voxel_indices_xyz"]):
        raise RuntimeError("FMT and proxy voxel indices differ")
    if not (len(seeds_xyz) == len(vortex_ids) == len(proxy_classes) == len(angles)):
        raise RuntimeError("frozen source arrays have inconsistent lengths")

    flow_path = _resolve_flow(config["input"]["flow_path_candidates"], args.flow)
    field, flow_metadata = ChannelVelocityField3D.from_vtk(flow_path)
    resolution_xyz = np.asarray(source["resolution_xyz"], dtype=np.int32)
    domain_min_xyz = np.asarray(source["domain_min_xyz"], dtype=float)
    domain_max_xyz = np.asarray(source["domain_max_xyz"], dtype=float)
    voxel_size_xyz = (domain_max_xyz - domain_min_xyz) / resolution_xyz
    mean_voxel_size = float(np.mean(voxel_size_xyz))
    minimum_voxel_size = float(np.min(voxel_size_xyz))

    encoder = config["encoder"]
    base_width = fmt_base_feature_width(
        int(encoder["num_freq"]),
        str(encoder["mode"]),
        bool(encoder["include_chirality"]),
    )
    streamline_grid = config["streamline_search"]
    phase1_kmeans = config["phase1_kmeans"]
    topology_config = config["topology_naming"]
    search_rows: list[dict] = []
    feature_cache: dict[tuple[float, int], dict] = {}

    for step_scale in streamline_grid["spatial_step_mean_voxel_scales"]:
        for steps in streamline_grid["steps_per_direction"]:
            step_scale = float(step_scale)
            steps = int(steps)
            primitives, valid = integrate_bidirectional_cross_primitives(
                field,
                seeds_xyz,
                spatial_step=step_scale * mean_voxel_size,
                steps_per_direction=steps,
                offset=float(streamline_grid["offset_min_voxel_scale"])
                * minimum_voxel_size,
                minimum_speed=float(streamline_grid["minimum_speed"]),
                chunk_size=int(streamline_grid["chunk_size"]),
            )
            if any(np.count_nonzero(valid & (vortex_ids == vortex_id)) < 2
                   for vortex_id in np.unique(vortex_ids)):
                raise RuntimeError(
                    f"candidate step={step_scale}, steps={steps} leaves an instance "
                    "with fewer than two valid primitives"
                )
            raw_features = pathline_dft_features_3d(
                torch.from_numpy(primitives[valid]),
                num_freq=int(encoder["num_freq"]),
                neighbor_weight=1.0,
                neighbor_scale=1.0,
                neighbor_pool=str(encoder["neighbor_pool"]),
                mode=str(encoder["mode"]),
                include_chirality=bool(encoder["include_chirality"]),
            ).astype(np.float32, copy=False)
            feature_cache[(step_scale, steps)] = {
                "raw_features": raw_features,
                "valid": valid,
                "center_streamlines": primitives[:, 0].copy(),
            }
            for neighbor_weight in encoder["neighbor_weights_after_standardization"]:
                raw_clusters, _ = per_vortex_kmeans(
                    raw_features,
                    vortex_ids[valid],
                    base_feature_width=base_width,
                    neighbor_weight=float(neighbor_weight),
                    seed=int(config["seed"]),
                    **phase1_kmeans,
                )
                _, summaries = _evaluate_partition(
                    raw_clusters,
                    valid,
                    vortex_ids,
                    indices_xyz,
                    proxy_classes,
                    angles,
                    vorticity,
                    topology_config,
                )
                search_rows.append(
                    _candidate_row(
                        phase="phase1_streamline_encoder",
                        step_scale=step_scale,
                        steps=steps,
                        neighbor_weight=float(neighbor_weight),
                        kmeans_id="phase1_fixed",
                        kmeans_config=phase1_kmeans,
                        valid=valid,
                        raw_clusters_valid=raw_clusters,
                        proxy_classes=proxy_classes,
                        vortex_ids=vortex_ids,
                        summaries=summaries,
                    )
                )
            _write_csv(output_dir / "search_progress.csv", search_rows)
            print(
                json.dumps(
                    {
                        "completed_representation": [step_scale, steps],
                        "candidate_count": len(search_rows),
                        "best_proxy_permutation_macro_f1": max(
                            row["proxy_permutation_macro_f1"] for row in search_rows
                        ),
                    }
                ),
                flush=True,
            )

    phase1_rows = [row for row in search_rows if row["phase"].startswith("phase1")]
    phase1_winner = max(phase1_rows, key=_selection_key)
    winning_key = (
        float(phase1_winner["spatial_step_mean_voxel_scale"]),
        int(phase1_winner["steps_per_direction"]),
    )
    winning_features = feature_cache[winning_key]
    valid = winning_features["valid"]
    for variant in config["phase2_kmeans_variants"]:
        variant = dict(variant)
        variant_id = str(variant.pop("id"))
        raw_clusters, _ = per_vortex_kmeans(
            winning_features["raw_features"],
            vortex_ids[valid],
            base_feature_width=base_width,
            neighbor_weight=float(
                phase1_winner["neighbor_weight_after_standardization"]
            ),
            seed=int(config["seed"]),
            **variant,
        )
        details, summaries = _evaluate_partition(
            raw_clusters,
            valid,
            vortex_ids,
            indices_xyz,
            proxy_classes,
            angles,
            vorticity,
            topology_config,
        )
        search_rows.append(
            _candidate_row(
                phase="phase2_kmeans",
                step_scale=winning_key[0],
                steps=winning_key[1],
                neighbor_weight=float(
                    phase1_winner["neighbor_weight_after_standardization"]
                ),
                kmeans_id=variant_id,
                kmeans_config=variant,
                valid=valid,
                raw_clusters_valid=raw_clusters,
                proxy_classes=proxy_classes,
                vortex_ids=vortex_ids,
                summaries=summaries,
            )
        )
    search_path = output_dir / "parameter_search.csv"
    _write_csv(search_path, search_rows)

    phase2_rows = [row for row in search_rows if row["phase"] == "phase2_kmeans"]
    winner = max(phase2_rows, key=_selection_key)
    winner_variant = next(
        dict(item) for item in config["phase2_kmeans_variants"]
        if item["id"] == winner["kmeans_id"]
    )
    winner_variant.pop("id")
    raw_clusters, kmeans_rows = per_vortex_kmeans(
        winning_features["raw_features"],
        vortex_ids[valid],
        base_feature_width=base_width,
        neighbor_weight=float(winner["neighbor_weight_after_standardization"]),
        seed=int(config["seed"]),
        **winner_variant,
    )
    final_details, final_summaries = _evaluate_partition(
        raw_clusters,
        valid,
        vortex_ids,
        indices_xyz,
        proxy_classes,
        angles,
        vorticity,
        topology_config,
    )

    per_mode_paths = {}
    for mode, rows in final_details["per_vortex_metrics"].items():
        path = output_dir / f"per_vortex_proxy_metrics_{mode}.csv"
        _write_csv(path, rows)
        per_mode_paths[mode] = str(path.resolve())
    for mode, rows in final_details["mapping_rows"].items():
        _write_csv(output_dir / f"per_vortex_mapping_{mode}.csv", rows)
    _write_csv(output_dir / "per_vortex_kmeans_diagnostics.csv", kmeans_rows)

    result_path = output_dir / "per_vortex_fmt_proxy_sweep_result.npz"
    np.savez_compressed(
        result_path,
        resolution_xyz=resolution_xyz,
        domain_min_xyz=domain_min_xyz,
        domain_max_xyz=domain_max_xyz,
        voxel_indices_xyz=indices_xyz,
        seeds_xyz=seeds_xyz.astype(np.float32),
        vortex_ids=vortex_ids,
        primitive_valid=valid,
        center_streamlines=winning_features["center_streamlines"],
        raw_fmt_features_valid=winning_features["raw_features"],
        raw_clusters=final_details["raw_clusters_full"],
        semantic_classes_proxy_permutation=final_details["semantic_classes"][
            "proxy_permutation"
        ],
        semantic_classes_vorticity_axis=final_details["semantic_classes"][
            "vorticity_axis"
        ],
        semantic_classes_hairpin_topology=final_details["semantic_classes"][
            "hairpin_topology"
        ],
        proxy_classes=proxy_classes,
        proxy_angle_degrees=angles,
    )

    selection = config["selection"]
    oracle = final_summaries["proxy_permutation"]
    macro_f1 = float(oracle["primary_vortex_equal_weighted_macro_f1"]["mean"])
    hard_mse = float(oracle["vortex_equal_weighted_hard_label_mse"]["mean"])
    summary = {
        "experiment": config["experiment"],
        "task": config["task"],
        "config": str(config_path.resolve()),
        "source_fmt_result": str(source_path.resolve()),
        "source_proxy_result": str(proxy_path.resolve()),
        "flow": flow_metadata,
        "candidate_count": len(search_rows),
        "phase1_winner": phase1_winner,
        "winner": winner,
        "final_proxy_metrics_by_semantic_mapping": final_summaries,
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
        "selection_boundary": (
            "The proxy-permutation score is an external clustering upper bound: "
            "velocity-curl labels select the label swap independently within each "
            "VortexId. Hairpin-topology naming is the deployable geometry-only rule."
        ),
        "parameter_search_path": str(search_path.resolve()),
        "per_vortex_metric_paths": per_mode_paths,
        "result_path": str(result_path.resolve()),
    }
    summary_path = output_dir / "summary.json"
    summary["summary_path"] = str(summary_path.resolve())
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
