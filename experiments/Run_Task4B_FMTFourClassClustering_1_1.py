"""Run the frozen Task1-FMT KMeans(k=4) Task4-b diagnostic."""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yaml

from FMT_Utils.Task4A_PerVortexClustering_3D import fmt_base_feature_width
from FMT_Utils.Task4A_StreamlineClustering_3D import (
    summarize_topology,
    topology_proxy_rows,
)
from FMT_Utils.Task4B_ProxyLabels_3D import (
    CLASS_NAMES,
    HAIRPIN_HEAD,
    HAIRPIN_LIMB,
    apply_cluster_mapping,
    best_cluster_permutation,
    four_class_metrics,
)
from FMT_Utils.VoxelSegmentation_3D import (
    SegmentationField3D,
    VoxelGrid3D,
    render_voxel_segmentation,
)


DEFAULT_CONFIG = Path("config/Other_Task4B_FMTFourClassClustering_1.2.yaml")
SPLIT_NAMES = ("train", "validation", "test")
SHORT_CLASS_NAMES = (
    "Ordinary\nstreamwise",
    "Ordinary\nspanwise",
    "Hairpin\nhead",
    "Hairpin\nlimb",
)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _class_color(class_id: int) -> tuple[float, float, float]:
    """Match VoxelSegmentation_3D's deterministic color for class_id+1."""

    hue = (0.6180339887498949 * (int(class_id) + 1)) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.68, 0.95)


def _per_vortex_hairpin_rows(
    labels: np.ndarray,
    predicted: np.ndarray,
    vortex_ids: np.ndarray,
    split_codes: np.ndarray,
) -> list[dict]:
    rows = []
    for vortex_id in np.unique(vortex_ids[vortex_ids > 0]):
        member = vortex_ids == vortex_id
        split_values = np.unique(split_codes[member])
        if len(split_values) != 1:
            raise RuntimeError(f"VortexId {int(vortex_id)} crosses splits")
        truth = labels[member]
        guess = predicted[member]
        head_truth = truth == HAIRPIN_HEAD
        limb_truth = truth == HAIRPIN_LIMB

        def class_f1(label: int) -> float:
            true_mask = truth == label
            pred_mask = guess == label
            tp = int(np.count_nonzero(true_mask & pred_mask))
            fp = int(np.count_nonzero(~true_mask & pred_mask))
            fn = int(np.count_nonzero(true_mask & ~pred_mask))
            denominator = 2 * tp + fp + fn
            return float(0.0 if denominator == 0 else 2.0 * tp / denominator)

        head_f1 = class_f1(HAIRPIN_HEAD)
        limb_f1 = class_f1(HAIRPIN_LIMB)
        rows.append(
            {
                "vortex_id": int(vortex_id),
                "split": SPLIT_NAMES[int(split_values[0])],
                "voxel_count": int(np.count_nonzero(member)),
                "head_count": int(np.count_nonzero(head_truth)),
                "limb_count": int(np.count_nonzero(limb_truth)),
                "head_f1": head_f1,
                "limb_f1": limb_f1,
                "head_limb_macro_f1": float(0.5 * (head_f1 + limb_f1)),
                "predicted_as_ordinary_count": int(
                    np.count_nonzero(member & np.isin(predicted, (0, 1)))
                ),
            }
        )
    return rows


def _topology_rows(
    resolution_xyz: np.ndarray,
    indices_xyz: np.ndarray,
    vortex_ids: np.ndarray,
    four_classes: np.ndarray,
) -> tuple[list[dict], dict]:
    nx, ny, nz = (int(value) for value in resolution_xyz)
    instances = np.zeros((nz, ny, nx), dtype=np.int32)
    types = np.zeros_like(instances, dtype=np.int8)
    ix, iy, iz = indices_xyz.T
    hairpin = vortex_ids > 0
    instances[iz[hairpin], iy[hairpin], ix[hairpin]] = vortex_ids[hairpin]
    semantic = np.zeros(len(four_classes), dtype=np.int8)
    semantic[four_classes == HAIRPIN_LIMB] = 1
    semantic[four_classes == HAIRPIN_HEAD] = 2
    types[iz[hairpin], iy[hairpin], ix[hairpin]] = semantic[hairpin]
    rows = topology_proxy_rows(
        instances,
        types,
        dominant_min_voxels=3,
        dominant_min_fraction=0.05,
    )
    return rows, summarize_topology(rows)


def _render_volumes(
    output_dir: Path,
    config: dict,
    resolution_xyz: np.ndarray,
    domain_min_xyz: np.ndarray,
    domain_max_xyz: np.ndarray,
    indices_xyz: np.ndarray,
    vortex_ids: np.ndarray,
    split_codes: np.ndarray,
    labels: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    grid = VoxelGrid3D(
        tuple(int(value) for value in resolution_xyz),
        domain_min_xyz,
        domain_max_xyz,
    )
    ix, iy, iz = indices_xyz.T
    proxy = np.zeros(grid.shape_zyx, dtype=np.int8)
    kmeans = np.zeros_like(proxy)
    proxy[iz, iy, ix] = labels + 1
    kmeans[iz, iy, ix] = predicted + 1
    hairpin = (vortex_ids > 0) & (split_codes == 2)
    proxy_hairpin = np.zeros_like(proxy)
    kmeans_hairpin = np.zeros_like(proxy)
    proxy_hairpin[iz[hairpin], iy[hairpin], ix[hairpin]] = labels[hairpin] + 1
    kmeans_hairpin[iz[hairpin], iy[hairpin], ix[hairpin]] = predicted[hairpin] + 1
    visual = config["visualization"]
    common = {
        "views": visual["views"],
        "surface_mode": str(visual["surface_mode"]),
        "shrink": float(visual["shrink"]),
        "opacity": float(visual["opacity"]),
        "image_size": visual["image_size"],
        "interactive": False,
    }
    outputs = {}
    for key, volume, name in (
        ("proxy", proxy, "task4b_proxy_sampled_support"),
        ("kmeans", kmeans, "task4b_fmt_kmeans_sampled_support"),
        ("proxy_hairpin", proxy_hairpin, "task4b_proxy_test_hairpin_only"),
        ("kmeans_hairpin", kmeans_hairpin, "task4b_fmt_kmeans_test_hairpin_only"),
    ):
        outputs[key] = render_voxel_segmentation(
            SegmentationField3D(grid, volume, name=name),
            output_dir=output_dir / f"{key}_views",
            **common,
        )
    return outputs


def _find_view(render: dict, suffix: str) -> Path:
    for value in render["view_paths"]:
        path = Path(value)
        if path.stem.endswith(suffix):
            return path
    raise FileNotFoundError(f"render has no view ending {suffix!r}")


def _make_summary_figure(
    output_dir: Path,
    renders: dict,
    test_metrics: dict,
) -> dict:
    """Image plate plus quantification for one bounded failure claim."""

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )
    proxy_image = plt.imread(_find_view(renders["proxy_hairpin"], "isometric"))
    kmeans_image = plt.imread(_find_view(renders["kmeans_hairpin"], "isometric"))
    matrix = np.asarray(
        test_metrics["confusion_matrix_true_rows_predicted_columns"],
        dtype=np.float64,
    )
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
    per_f1 = np.asarray(
        [test_metrics["per_class_f1"][name] for name in CLASS_NAMES]
    )
    colors = [_class_color(index) for index in range(4)]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    fig.subplots_adjust(
        left=0.085, right=0.985, bottom=0.10, top=0.95,
        wspace=0.22, hspace=0.30,
    )
    panels = axes.ravel()
    for ax, image, title in (
        (panels[0], proxy_image, "Proxy anatomy — test hairpins"),
        (panels[1], kmeans_image, "Mapped FMT KMeans — test hairpins"),
    ):
        ax.imshow(image, aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=name.replace("\n", " "))
        for color, name in zip(colors, SHORT_CLASS_NAMES)
    ]
    panels[1].legend(
        handles=legend_handles,
        loc="lower right",
        ncol=2,
        fontsize=5.8,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.4,
        borderpad=0.35,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="0.75",
        title="Class colors",
        title_fontsize=6.0,
    )

    heat = panels[2].imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    del heat
    for row in range(4):
        for column in range(4):
            value = normalized[row, column]
            panels[2].text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.52 else "#202020",
                fontsize=6.5,
            )
    panels[2].set(
        xticks=np.arange(4),
        yticks=np.arange(4),
        xticklabels=SHORT_CLASS_NAMES,
        yticklabels=SHORT_CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="Proxy class",
        title="Test confusion (row-normalized)",
    )
    bars = panels[3].bar(np.arange(4), per_f1, color=colors, width=0.72)
    for bar, value in zip(bars, per_f1):
        panels[3].text(
            bar.get_x() + 0.5 * bar.get_width(),
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    panels[3].axhline(
        test_metrics["macro_f1"], color="#272727", linestyle="--", linewidth=1.0,
    )
    panels[3].set(
        xticks=np.arange(4),
        xticklabels=SHORT_CLASS_NAMES,
        ylabel="F1 score",
        ylim=(0.0, 1.08),
        title="Class-specific test performance",
    )
    panels[3].text(
        0.98,
        0.91,
        f"Macro-F1 = {test_metrics['macro_f1']:.2f}",
        transform=panels[3].transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
    )
    panels[3].grid(axis="y", color="0.90", linewidth=0.6)

    skill_scripts = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    from matplotlib.transforms import ScaledTranslation

    for label, ax in zip("abcd", panels):
        subplot_box = ax.get_subplotspec().get_position(fig)
        offset = ScaledTranslation(
            -9.0 / 72.0, 4.0 / 72.0, fig.dpi_scale_trans
        )
        ax.text(
            subplot_box.x0,
            ax.get_position().y1,
            label,
            transform=fig.transFigure + offset,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    fig.canvas.draw()
    qa_dir = output_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    alignment = require_matplotlib_panel_alignment(
        fig,
        axes=list(panels),
        panel_ids=list("abcd"),
        row_groups=[["a", "b"], ["c", "d"]],
        # Image plates (a,b) and quantitative axes (c,d) have intentionally
        # different label/tick margins, so only within-row plot areas are
        # scientifically comparable.
        column_groups=[],
        exemptions=[
            {
                "panels": ["c", "d"],
                "checks": ["column"],
                "reason": (
                    "Quantitative panels carry axis/tick margins while image "
                    "plates do not; only within-row plot areas are comparable."
                ),
            }
        ],
        json_out=qa_dir / "panel_alignment.json",
        overlay_svg=qa_dir / "panel_alignment_overlay.svg",
        require_panel_labels=True,
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    stem = output_dir / "task4b_fmt_four_class_kmeans_summary"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    tiff_path = stem.with_suffix(".tiff")
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    fig.savefig(
        tiff_path,
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    paths = {
        "png": str(png_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "svg": str(svg_path.resolve()),
        "tiff": str(tiff_path.resolve()),
    }
    plt.close(fig)
    return {
        "paths": paths,
        "alignment_verdict": alignment.get("verdict"),
        "alignment_json": str((qa_dir / "panel_alignment.json").resolve()),
    }


def run(
    config_path: str | Path,
    *,
    output_override: str | None = None,
    no_images: bool = False,
) -> Path:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cache_path = Path(spec["cache"]["path"])
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    output_dir = Path(output_override or spec["visualization"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "config_snapshot.yaml"
    if snapshot.exists():
        old = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        if json.loads(json.dumps(old, sort_keys=True)) != json.loads(
            json.dumps(spec, sort_keys=True)
        ):
            raise RuntimeError("clustering config changed; use a new experiment version")
    snapshot.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    with np.load(cache_path) as cache:
        features = np.asarray(cache["fmt_features"], dtype=np.float64)
        labels = np.asarray(cache["labels"], dtype=np.int8)
        split_codes = np.asarray(cache["split_codes"], dtype=np.int8)
        seeds = np.asarray(cache["seeds_xyz"], dtype=np.float32)
        vortex_ids = np.asarray(cache["vortex_ids"], dtype=np.int32)
        indices_xyz = np.asarray(cache["voxel_indices_xyz"], dtype=np.int32)
        resolution_xyz = np.asarray(cache["resolution_xyz"], dtype=np.int32)
        domain_min_xyz = np.asarray(cache["domain_min_xyz"], dtype=np.float64)
        domain_max_xyz = np.asarray(cache["domain_max_xyz"], dtype=np.float64)

    train = split_codes == 0
    validation = split_codes == 1
    test = split_codes == 2
    scaler = StandardScaler().fit(features[train])
    transformed = scaler.transform(features)
    encoder = spec["encoder"]
    base_width = fmt_base_feature_width(
        int(encoder["num_freq"]),
        str(encoder["mode"]),
        bool(encoder["include_chirality"]),
    )
    transformed[:, base_width:] *= float(
        encoder["neighbor_weight_after_train_standardization"]
    )
    cluster_spec = spec["clustering"]
    model = KMeans(
        n_clusters=int(cluster_spec["n_clusters"]),
        init=str(cluster_spec["init"]),
        n_init=int(cluster_spec["n_init"]),
        max_iter=int(cluster_spec["max_iter"]),
        tol=float(cluster_spec["tol"]),
        algorithm=str(cluster_spec["algorithm"]),
        random_state=int(spec["seed"]),
    ).fit(transformed[train])
    raw_clusters = model.predict(transformed).astype(np.int8)
    mapping, validation_mapping_f1 = best_cluster_permutation(
        labels[validation],
        raw_clusters[validation],
        cluster_ids_universe=tuple(range(int(cluster_spec["n_clusters"]))),
    )
    predicted = apply_cluster_mapping(raw_clusters, mapping)
    oracle_mapping, oracle_mapping_f1 = best_cluster_permutation(
        labels[test],
        raw_clusters[test],
        cluster_ids_universe=tuple(range(int(cluster_spec["n_clusters"]))),
    )
    oracle_predicted = apply_cluster_mapping(raw_clusters[test], oracle_mapping)

    metrics_by_split = {
        name: four_class_metrics(labels[split_codes == code], predicted[split_codes == code])
        for code, name in enumerate(SPLIT_NAMES)
    }
    test_oracle_metrics = four_class_metrics(labels[test], oracle_predicted)
    metric_rows = []
    for split_name, metrics in metrics_by_split.items():
        row = {
            "split": split_name,
            "sample_count": metrics["sample_count"],
            "macro_f1": metrics["macro_f1"],
            "macro_iou": metrics["macro_iou"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "adjusted_rand_index": metrics["adjusted_rand_index"],
            "normalized_mutual_information": metrics[
                "normalized_mutual_information"
            ],
        }
        for class_name in CLASS_NAMES:
            row[f"f1_{class_name}"] = metrics["per_class_f1"][class_name]
            row[f"support_{class_name}"] = metrics["support"][class_name]
        metric_rows.append(row)
    _write_csv(output_dir / "split_metrics.csv", metric_rows)

    per_vortex_rows = _per_vortex_hairpin_rows(
        labels, predicted, vortex_ids, split_codes
    )
    _write_csv(output_dir / "per_vortex_hairpin_metrics.csv", per_vortex_rows)
    proxy_topology_rows, proxy_topology = _topology_rows(
        resolution_xyz, indices_xyz, vortex_ids, labels
    )
    predicted_topology_rows, predicted_topology = _topology_rows(
        resolution_xyz, indices_xyz, vortex_ids, predicted
    )
    split_by_vortex_id = {
        int(row["vortex_id"]): str(row["split"]) for row in per_vortex_rows
    }
    for rows in (proxy_topology_rows, predicted_topology_rows):
        for row in rows:
            row["split"] = split_by_vortex_id[int(row["vortex_id"])]
    proxy_test_topology = summarize_topology(
        [row for row in proxy_topology_rows if row["split"] == "test"]
    )
    predicted_test_topology = summarize_topology(
        [row for row in predicted_topology_rows if row["split"] == "test"]
    )
    _write_csv(output_dir / "proxy_topology.csv", proxy_topology_rows)
    _write_csv(output_dir / "kmeans_topology.csv", predicted_topology_rows)

    result_path = output_dir / "task4b_fmt_four_class_kmeans_result.npz"
    np.savez_compressed(
        result_path,
        labels=labels,
        raw_clusters=raw_clusters,
        predicted_classes=predicted,
        split_codes=split_codes,
        seeds_xyz=seeds,
        vortex_ids=vortex_ids,
        voxel_indices_xyz=indices_xyz,
        resolution_xyz=resolution_xyz,
        domain_min_xyz=domain_min_xyz,
        domain_max_xyz=domain_max_xyz,
        cluster_centers=model.cluster_centers_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )

    renders = {}
    figure = {}
    if not no_images:
        renders = _render_volumes(
            output_dir,
            spec,
            resolution_xyz,
            domain_min_xyz,
            domain_max_xyz,
            indices_xyz,
            vortex_ids,
            split_codes,
            labels,
            predicted,
        )
        figure = _make_summary_figure(
            output_dir, renders, metrics_by_split["test"]
        )

    test_vortex_rows = [row for row in per_vortex_rows if row["split"] == "test"]
    summary = {
        "experiment": spec["experiment"],
        "config": str(config_path.resolve()),
        "cache": str(cache_path.resolve()),
        "result": str(result_path.resolve()),
        "method": {
            "feature": "frozen Task1 FMT",
            "n_clusters": 4,
            "train_sample_count": int(np.count_nonzero(train)),
            "validation_mapping": {str(k): int(v) for k, v in mapping.items()},
            "validation_mapping_macro_f1": validation_mapping_f1,
            "kmeans_inertia": float(model.inertia_),
            "kmeans_iterations": int(model.n_iter_),
        },
        "metrics": metrics_by_split,
        "test_oracle_mapping_report_only": {
            "mapping": {str(k): int(v) for k, v in oracle_mapping.items()},
            "permutation_selection_macro_f1": oracle_mapping_f1,
            "metrics": test_oracle_metrics,
        },
        "test_hairpin_vortex_equal_head_limb_macro_f1": {
            "mean": float(np.mean([row["head_limb_macro_f1"] for row in test_vortex_rows])),
            "median": float(np.median([row["head_limb_macro_f1"] for row in test_vortex_rows])),
            "vortex_id_count": len(test_vortex_rows),
        },
        "topology": {
            "proxy": proxy_topology,
            "mapped_kmeans": predicted_topology,
        },
        "topology_test_only": {
            "proxy": proxy_test_topology,
            "mapped_kmeans": predicted_test_topology,
        },
        "figure_contract": {
            "core_conclusion": (
                "Frozen rotation-invariant Task1 FMT KMeans does not reliably "
                "recover the coordinate-dependent four-class channel proxy."
            ),
            "results_question": (
                "Can label-free KMeans on frozen Task1 FMT separate ordinary "
                "streamwise/spanwise vortices and hairpin head/limb?"
            ),
            "archetype": "image plate + quant",
            "backend": "Python/matplotlib plus VTK cube rendering",
            "panel_map": {
                "a": "proxy hairpin anatomy",
                "b": "mapped KMeans hairpin segmentation",
                "c": "test confusion matrix",
                "d": "per-class test F1",
            },
            "reviewer_risk": (
                "Proxy labels are not manual anatomy; one steady channel supplies "
                "spatial holdout but not temporal or cross-flow confirmation."
            ),
        },
        "render": renders,
        "figure": figure,
        "interpretation_boundary": (
            "Validation labels only name arbitrary clusters; test labels do not "
            "fit KMeans or the deployed mapping. The separately reported test "
            "oracle is non-deployable."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(metric_rows, indent=2), flush=True)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    parser.add_argument("--no-images", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.config, output_override=args.output_dir, no_images=args.no_images)
