"""mainExp_3DFMT_1.1: training-free 3D pathline clustering baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from DeepUtils.utils import EasyConfig
from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.FMT_3D_pipeline import (
    generate_seeding_grid_3d,
    compute_ivd_reference_3d,
    integrate_cross_primitives_3d,
    load_vector_field_3d,
    visualize_3d_clustering,
    visualize_ivd_reference_3d,
    write_run_metadata,
)


EXPERIMENT_VERSION = "mainExp_3DFMT_1.1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/PathlineFMTclustering3D.yaml")
    parser.add_argument("--input", help="override dataset.path with a .nc or .npz file")
    parser.add_argument("--output", help="override output directory")
    return parser.parse_args()


def run(config, input_override=None, output_override=None):
    seed = int(config.seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    input_path = Path(input_override or config.dataset.path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(
            f"3D vector field not found: {input_path}. Set dataset.path or pass --input."
        )
    output_dir = Path(output_override or config.output.dir) / EXPERIMENT_VERSION / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    field = load_vector_field_3d(input_path)
    seed_time = field.tmin + float(config.dataset.seed_time_ratio) * (field.tmax - field.tmin)
    dt = float(field.timeInterval) * float(config.pathlines.dt_scale)
    if dt <= 0:
        raise ValueError("unsteady 3D data with at least two distinct time samples is required")
    offset = float(np.min(field.gridInterval[field.gridInterval > 0])) * float(
        config.pathlines.offset_grid_scale
    )
    seeds, axes = generate_seeding_grid_3d(
        field,
        config.dataset.grid_shape,
        config.dataset.boundary_fraction,
        offset,
    )
    primitives, valid_mask, lengths = integrate_cross_primitives_3d(
        field,
        seeds,
        seed_time,
        dt,
        int(config.pathlines.integration_steps),
        int(config.pathlines.sampled_steps),
        offset,
        method=str(config.pathlines.method),
        chunk_size=int(config.pathlines.chunk_size),
    )
    seeds_valid = seeds[valid_mask]
    if len(seeds_valid) < int(config.clustering.classes):
        raise RuntimeError(
            f"only {len(seeds_valid)} complete primitives remain; cannot cluster. "
            "Reduce integration_steps or increase boundary_fraction."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    primitive_tensor = torch.from_numpy(primitives).to(device)
    features = pathline_dft_features_3d(
        primitive_tensor,
        num_freq=int(config.encoder.num_freq),
        # Apply this weight after StandardScaler; applying it here would be
        # cancelled exactly by per-column standardization.
        neighbor_weight=1.0,
        # A uniform pre-DFT neighbour scale is also cancelled by StandardScaler.
        neighbor_scale=1.0,
        neighbor_pool=str(config.encoder.neighbor_pool),
        mode=str(config.encoder.mode),
        include_chirality=bool(config.encoder.include_chirality),
    )
    if not np.isfinite(features).all():
        raise RuntimeError("encoder produced NaN or infinite features")
    standardized = StandardScaler().fit_transform(features)
    base_width = int(config.encoder.num_freq) * (
        1 if str(config.encoder.mode) == "magnitude" else 3
    ) + (int(config.encoder.num_freq) - 1 if config.encoder.include_chirality else 0)
    standardized[:, base_width:] *= float(config.encoder.neighbor_weight)
    kmeans = KMeans(
        n_clusters=int(config.clustering.classes),
        random_state=seed,
        n_init=int(config.clustering.n_init),
    )
    labels = kmeans.fit_predict(standardized)

    ivd_volume, ivd_at_seeds, ivd_axes = compute_ivd_reference_3d(field, seed_time, seeds_valid)
    ivd_percentiles = tuple(float(value) for value in config.visualization.ivd_percentiles)
    ivd_metrics = {}
    for percentile in ivd_percentiles:
        threshold = float(np.percentile(ivd_volume, percentile))
        reference = ivd_at_seeds >= threshold
        candidates = []
        for cluster in range(int(config.clustering.classes)):
            prediction = labels == cluster
            candidates.append({
                "cluster_as_vortex": cluster,
                "f1": float(f1_score(reference, prediction, zero_division=0)),
                "precision": float(precision_score(reference, prediction, zero_division=0)),
                "recall": float(recall_score(reference, prediction, zero_division=0)),
                "iou": float(jaccard_score(reference, prediction, zero_division=0)),
            })
        best = max(candidates, key=lambda item: item["f1"])
        ivd_metrics[str(percentile)] = {
            "threshold": threshold,
            "positive_seed_count": int(reference.sum()),
            **best,
        }

    all_labels = np.full(len(seeds), -1, dtype=np.int8)
    all_labels[valid_mask] = labels.astype(np.int8)
    np.savez_compressed(
        output_dir / "clustering_result.npz",
        labels=labels,
        labels_full_grid=all_labels,
        seeds=seeds_valid,
        seeds_full_grid=seeds,
        valid_mask=valid_mask,
        pathlines=primitives,
        features=features,
        standardized_features=standardized,
        cluster_centers=kmeans.cluster_centers_,
        ivd_volume=ivd_volume,
        ivd_at_seeds=ivd_at_seeds,
        x=axes[0], y=axes[1], z=axes[2],
        line_lengths=lengths,
    )
    visualize_3d_clustering(
        seeds_valid, labels, primitives, output_dir,
        max_lines=int(config.visualization.max_pathlines),
    )
    ivd_levels = visualize_ivd_reference_3d(
        ivd_volume, ivd_axes, field.domainMinBoundary, field.gridInterval,
        seeds_valid, labels, output_dir, percentiles=ivd_percentiles,
    )
    counts = np.bincount(labels, minlength=int(config.clustering.classes))
    metadata = {
        "experiment": EXPERIMENT_VERSION,
        "input": str(input_path),
        "device": str(device),
        "field_shape_TZYXC": list(field.field.shape),
        "seed_time": float(seed_time),
        "dt": dt,
        "offset": offset,
        "total_primitives": int(len(seeds)),
        "valid_primitives": int(len(seeds_valid)),
        "feature_width": int(features.shape[1]),
        "cluster_counts": counts.tolist(),
        "ivd_isosurface_levels": ivd_levels,
        "ivd_cluster_metrics": ivd_metrics,
        "config": config.dict(),
        "note": "KMeans cluster IDs are arbitrary; this run does not name either cluster vortex.",
    }
    write_run_metadata(output_dir / "run_metadata.json", metadata)
    print(f"[{EXPERIMENT_VERSION}] output: {output_dir}")
    print(f"valid primitives: {len(seeds_valid)}/{len(seeds)}")
    print(f"feature shape: {features.shape}; cluster counts: {counts.tolist()}")
    for percentile, values in ivd_metrics.items():
        print(
            f"IVD p{percentile}: best cluster={values['cluster_as_vortex']}, "
            f"F1={values['f1']:.3f}, IoU={values['iou']:.3f}, "
            f"precision={values['precision']:.3f}, recall={values['recall']:.3f}"
        )
    return output_dir


if __name__ == "__main__":
    args = parse_args()
    run(EasyConfig(args.config), args.input, args.output)
