"""Static full-field TBL segmentation plates for Task4-b 2.3.

Figure contract
---------------
Core conclusion: compare the four-class TBL proxy segmentation with Raw,
Raw-wide, FMT-only, and Raw+FMT predictions from one explicitly selected
optimizer seed, without selecting or sampling test cubes.
Evidence: one five-panel plate per fixed orthographic view.  Every panel uses
the same ordered target rows, projected voxel positions, bounds, and class
colors.  The proxy is a physics-derived target, not manual anatomy truth.
Archetype: image plate plus machine-readable data-integrity validation.

Rendering uses a deterministic orthographic occupancy z-buffer.  All target
rows enter every view.  For each projected voxel ray, the nearest occupied
voxel is visible and the remaining rows are recorded as physically occluded;
no row is randomly sampled or silently discarded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
import numpy as np


EXPERIMENT = "mainExp_Task4B_ChannelToTBL_2.3"
VISUALIZATION_VERSION = "Visualize_Task4B_ChannelToTBL_2.3"
DEFAULT_OUTPUT_DIR = Path("outputs/mainExp_Task4B_ChannelToTBL_2.3/tbl_GTs")
DEFAULT_SEED = 7068
DEFAULT_VIEWS = ("+z", "+y", "+x")
VARIANTS = ("raw", "raw_wide", "fmt_only", "raw_fmt")
METHODS = ("proxy",) + VARIANTS
METHOD_LABELS = {
    "proxy": "Proxy target",
    "raw": "Raw",
    "raw_wide": "Raw-wide",
    "fmt_only": "FMT-only",
    "raw_fmt": "Raw+FMT",
}
CLASS_NAMES = (
    "ordinary_streamwise",
    "ordinary_spanwise",
    "hairpin_head",
    "hairpin_limb",
)
CLASS_LABELS = (
    "Ordinary streamwise",
    "Ordinary spanwise",
    "Hairpin head",
    "Hairpin limb",
)
# Identical to the frozen Task4-b 1.2 visualization palette.
CLASS_COLORS = ("#0072B2", "#E69F00", "#CC79A7", "#009E73")
PREDICTION_KEYS = {
    "source_row_indices",
    "source_targets",
    "source_probabilities",
    "source_predicted_labels",
    "target_source_candidate_indices",
    "target_voxel_indices_xyz",
    "target_seeds_xyz",
    "target_vortex_ids",
    "target_targets",
    "target_probabilities",
    "target_predicted_labels",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "Liberation Sans",
            "sans-serif",
        ],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


class VisualizationInputError(RuntimeError):
    """Raised when a formal artifact violates the frozen figure contract."""


@dataclass(frozen=True)
class ViewSpec:
    name: str
    horizontal_axis: int
    vertical_axis: int
    depth_axis: int
    front: str
    title: str


VIEW_SPECS = {
    "+z": ViewSpec("+z", 0, 1, 2, "maximum", "View from +z (x–y)"),
    "-z": ViewSpec("-z", 0, 1, 2, "minimum", "View from −z (x–y)"),
    "+y": ViewSpec("+y", 0, 2, 1, "maximum", "View from +y (x–z)"),
    "-y": ViewSpec("-y", 0, 2, 1, "minimum", "View from −y (x–z)"),
    "+x": ViewSpec("+x", 1, 2, 0, "maximum", "View from +x (y–z)"),
    "-x": ViewSpec("-x", 1, 2, 0, "minimum", "View from −x (y–z)"),
}
AXIS_LABELS = ("x (streamwise)", "y (spanwise)", "z (vertical)")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VisualizationInputError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    """Use the independent auditor's exact typed-array hash convention."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    _require(np.isin(values, np.arange(4)).all(), "labels contain an unknown class")
    counts = np.bincount(values, minlength=4)
    return {name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--audit-json",
        help="Defaults to OUTPUT_DIR/independent_audit.json.",
    )
    parser.add_argument(
        "--target-manifest",
        help="Defaults to OUTPUT_DIR/target_cache/target_cache_manifest.json.",
    )
    parser.add_argument(
        "--figure-dir",
        help="Defaults to OUTPUT_DIR/figures/task4b_channel_to_tbl_2.3_seedSEED.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--view",
        action="append",
        choices=tuple(VIEW_SPECS),
        help="Fixed orthographic view; repeat for multiple views.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--alignment-helper-dir",
        help="Directory containing audit_panel_alignment.py.",
    )
    return parser.parse_args()


def _load_alignment_helper(
    requested_dir: str | None,
) -> Callable[..., dict[str, Any]]:
    candidates: list[Path] = []
    if requested_dir:
        candidates.append(Path(requested_dir).expanduser())
    if os.environ.get("NATURE_FIGURE_SCRIPTS"):
        candidates.append(Path(os.environ["NATURE_FIGURE_SCRIPTS"]).expanduser())
    candidates.extend(
        [
            Path.home() / ".codex" / "skills" / "nature-figure" / "scripts",
            Path(__file__).resolve().parents[1]
            / ".codex"
            / "nature-figure"
            / "scripts",
        ]
    )
    for directory in candidates:
        source = directory.resolve() / "audit_panel_alignment.py"
        if not source.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "task4b_tbl_audit_panel_alignment", source
        )
        _require(spec is not None and spec.loader is not None, f"cannot import {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        helper = getattr(module, "require_matplotlib_panel_alignment", None)
        _require(callable(helper), f"alignment helper is missing in {source}")
        return helper
    raise VisualizationInputError(
        "audit_panel_alignment.py was not found; pass --alignment-helper-dir"
    )


def _read_prediction(path: Path) -> dict[str, np.ndarray]:
    _require(path.is_file(), f"prediction file is missing: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        _require(set(artifact.files) == PREDICTION_KEYS, f"unexpected keys: {path}")
        arrays = {key: np.asarray(artifact[key]).copy() for key in artifact.files}
    count = len(arrays["target_targets"])
    expected_vector_keys = (
        "target_source_candidate_indices",
        "target_vortex_ids",
        "target_targets",
        "target_predicted_labels",
    )
    for key in expected_vector_keys:
        _require(arrays[key].shape == (count,), f"{path.name}: wrong shape for {key}")
    _require(
        arrays["target_voxel_indices_xyz"].shape == (count, 3),
        f"{path.name}: wrong target voxel shape",
    )
    _require(
        arrays["target_seeds_xyz"].shape == (count, 3),
        f"{path.name}: wrong target coordinate shape",
    )
    probabilities = np.asarray(arrays["target_probabilities"], dtype=np.float64)
    _require(probabilities.shape == (count, 4), f"{path.name}: wrong probability shape")
    _require(np.isfinite(probabilities).all(), f"{path.name}: non-finite probability")
    _require(
        np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5),
        f"{path.name}: probability rows do not sum to one",
    )
    predicted = np.asarray(arrays["target_predicted_labels"], dtype=np.int64)
    _require(
        np.array_equal(predicted, np.argmax(probabilities, axis=1)),
        f"{path.name}: saved labels differ from probability argmax",
    )
    _class_counts(arrays["target_targets"])
    _class_counts(predicted)
    return arrays


def _load_formal_inputs(
    output_dir: Path,
    audit_path: Path,
    target_manifest_path: Path,
    seed: int,
) -> dict[str, Any]:
    _require(audit_path.is_file(), f"independent audit is missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _require(audit.get("status") == "PASS", "independent audit status is not PASS")
    _require(audit.get("experiment") == EXPERIMENT, "audit experiment differs")
    _require(target_manifest_path.is_file(), f"target manifest is missing: {target_manifest_path}")
    target_manifest_sha = _sha256(target_manifest_path)
    _require(
        audit.get("target_manifest", {}).get("sha256") == target_manifest_sha,
        "target manifest changed after the independent audit",
    )
    target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    _require(target_manifest.get("experiment") == EXPERIMENT, "target manifest differs")

    audited_runs = {
        (str(row.get("variant")), int(row.get("seed"))): row
        for row in audit.get("runs", ())
    }
    expected_run_keys = {(variant, seed) for variant in VARIANTS}
    _require(
        expected_run_keys.issubset(audited_runs),
        f"seed {seed} is not independently audited for every variant",
    )

    predictions: dict[str, np.ndarray] = {}
    prediction_records: dict[str, Any] = {}
    canonical: dict[str, np.ndarray] | None = None
    row_keys = (
        "target_source_candidate_indices",
        "target_voxel_indices_xyz",
        "target_seeds_xyz",
        "target_vortex_ids",
        "target_targets",
    )
    ordered_row_sha: str | None = None
    for variant in VARIANTS:
        path = output_dir / "predictions" / f"{variant}_seed{seed}.npz"
        actual_sha = _sha256(path)
        audit_row = audited_runs[(variant, seed)]
        _require(
            actual_sha == audit_row.get("prediction_sha256"),
            f"prediction changed after audit: {path.name}",
        )
        arrays = _read_prediction(path)
        if canonical is None:
            canonical = {key: arrays[key] for key in row_keys}
            ordered_row_sha = _array_sha256(*(canonical[key] for key in row_keys))
        else:
            for key in row_keys:
                _require(
                    np.array_equal(arrays[key], canonical[key]),
                    f"{path.name}: ordered target rows differ for {key}",
                )
        _require(
            audit_row.get("target_ordered_row_sha256") == ordered_row_sha,
            f"{path.name}: ordered-row hash differs from audit",
        )
        predictions[variant] = np.asarray(
            arrays["target_predicted_labels"], dtype=np.int8
        )
        prediction_records[variant] = {
            "path": str(path.resolve()),
            "sha256": actual_sha,
            "seed": int(seed),
            "ordered_target_row_sha256": ordered_row_sha,
        }

    _require(canonical is not None and ordered_row_sha is not None, "no predictions loaded")
    target_evidence = audit.get("target_rows", {})
    _require(
        target_evidence.get("ordered_test_row_sha256") == ordered_row_sha,
        "ordered target rows differ from the independently audited target cache",
    )
    row_count = len(canonical["target_targets"])
    _require(
        int(target_evidence.get("row_count", -1)) == row_count,
        "target row count differs from audit",
    )

    grid = target_manifest.get("target_grid", {})
    resolution = np.asarray(grid.get("resolution_xyz"), dtype=np.int64)
    domain_min = np.asarray(grid.get("domain_min_xyz"), dtype=np.float64)
    domain_max = np.asarray(grid.get("domain_max_xyz"), dtype=np.float64)
    voxel_size = np.asarray(grid.get("voxel_size_xyz"), dtype=np.float64)
    _require(
        resolution.shape == domain_min.shape == domain_max.shape == voxel_size.shape == (3,),
        "target grid metadata is malformed",
    )
    voxels = np.asarray(canonical["target_voxel_indices_xyz"], dtype=np.int64)
    _require(
        np.all((voxels >= 0) & (voxels < resolution[None, :])),
        "target voxel indices lie outside the target grid",
    )
    linear = (voxels[:, 2] * resolution[1] + voxels[:, 1]) * resolution[0] + voxels[:, 0]
    _require(len(np.unique(linear)) == row_count, "target voxel rows are not unique")
    candidate = np.asarray(canonical["target_source_candidate_indices"], dtype=np.int64)
    _require(np.all(np.diff(candidate) > 0), "target candidate rows are reordered or repeated")
    expected_seeds = (
        domain_min[None, :] + (voxels.astype(np.float64) + 0.5) * voxel_size[None, :]
    ).astype(np.float32)
    _require(
        np.array_equal(canonical["target_seeds_xyz"], expected_seeds),
        "target coordinates do not exactly map to voxel centers",
    )

    labels = {"proxy": np.asarray(canonical["target_targets"], dtype=np.int8)}
    labels.update(predictions)
    vortex_ids = np.asarray(canonical["target_vortex_ids"], dtype=np.int32)
    _require(
        np.array_equal(vortex_ids > 0, labels["proxy"] >= 2),
        "manual hairpin support differs from proxy head/limb support",
    )
    return {
        "labels": labels,
        "voxels": voxels,
        "resolution": resolution,
        "domain_min": domain_min,
        "domain_max": domain_max,
        "voxel_size": voxel_size,
        "candidate_indices": candidate,
        "unique_voxel_count": int(len(np.unique(linear))),
        "vortex_ids": vortex_ids,
        "ordered_row_sha256": ordered_row_sha,
        "prediction_records": prediction_records,
        "audit": audit,
        "audit_sha256": _sha256(audit_path),
        "target_manifest_sha256": target_manifest_sha,
    }


def project_frontmost_rows(
    voxel_indices_xyz: np.ndarray,
    resolution_xyz: np.ndarray,
    view: str,
) -> dict[str, Any]:
    """Return one deterministic frontmost row per occupied projected ray.

    Every input row is assigned to exactly one ray.  ``visible_rows`` are the
    frontmost representatives; all other rows remain accounted for as
    ``occluded_row_count`` rather than being sampled away.
    """
    _require(view in VIEW_SPECS, f"unknown view: {view}")
    voxels = np.asarray(voxel_indices_xyz, dtype=np.int64)
    resolution = np.asarray(resolution_xyz, dtype=np.int64)
    _require(voxels.ndim == 2 and voxels.shape[1] == 3, "voxel array must be N x 3")
    _require(resolution.shape == (3,) and np.all(resolution > 0), "invalid resolution")
    _require(
        np.all((voxels >= 0) & (voxels < resolution[None, :])),
        "voxel indices lie outside the grid",
    )
    spec = VIEW_SPECS[view]
    horizontal = voxels[:, spec.horizontal_axis]
    vertical = voxels[:, spec.vertical_axis]
    depth = voxels[:, spec.depth_axis]
    width = int(resolution[spec.horizontal_axis])
    height = int(resolution[spec.vertical_axis])
    ray = vertical * width + horizontal
    depth_key = depth if spec.front == "maximum" else -depth
    row_number = np.arange(len(voxels), dtype=np.int64)
    order = np.lexsort((row_number, depth_key, ray))
    ordered_ray = ray[order]
    group_end = np.empty(len(order), dtype=bool)
    if len(order):
        group_end[:-1] = ordered_ray[:-1] != ordered_ray[1:]
        group_end[-1] = True
    visible_rows = order[group_end]
    visible_ray = ray[visible_rows]
    _require(len(np.unique(visible_ray)) == len(visible_rows), "z-buffer rays repeat")
    visible_count = int(len(visible_rows))
    input_count = int(len(voxels))
    return {
        "view": view,
        "visible_rows": visible_rows,
        "horizontal_indices": horizontal[visible_rows],
        "vertical_indices": vertical[visible_rows],
        "image_shape": (height, width),
        "input_row_count": input_count,
        "visible_row_count": visible_count,
        "occluded_row_count": input_count - visible_count,
        "excluded_row_count": 0,
        "all_input_rows_accounted_for": True,
    }


def labels_to_projected_image(
    labels: np.ndarray, projection: dict[str, Any]
) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int8).reshape(-1)
    _require(
        len(values) == int(projection["input_row_count"]),
        "label count differs from projection input",
    )
    _class_counts(values)
    image = np.full(tuple(projection["image_shape"]), -1, dtype=np.int8)
    rows = np.asarray(projection["visible_rows"], dtype=np.int64)
    horizontal = np.asarray(projection["horizontal_indices"], dtype=np.int64)
    vertical = np.asarray(projection["vertical_indices"], dtype=np.int64)
    image[vertical, horizontal] = values[rows]
    _require(
        int(np.count_nonzero(image >= 0)) == int(projection["visible_row_count"]),
        "projected image occupancy differs from z-buffer count",
    )
    return image


def true_hairpin_support_subset(data: dict[str, Any]) -> dict[str, Any]:
    """Crop all and only valid rows inside manual TBL hairpin support."""
    member = np.asarray(data["vortex_ids"], dtype=np.int64) > 0
    _require(np.any(member), "target contains no valid manual hairpin-support rows")
    proxy = np.asarray(data["labels"]["proxy"], dtype=np.int8)
    _require(
        np.array_equal(member, proxy >= 2),
        "hairpin-support selection differs from proxy head/limb rows",
    )
    global_voxels = np.asarray(data["voxels"], dtype=np.int64)[member]
    minimum = global_voxels.min(axis=0)
    maximum = global_voxels.max(axis=0)
    local_resolution = maximum - minimum + 1
    local_voxels = global_voxels - minimum[None, :]
    domain_min = data["domain_min"] + minimum * data["voxel_size"]
    domain_max = data["domain_min"] + (maximum + 1) * data["voxel_size"]
    labels = {method: values[member] for method, values in data["labels"].items()}
    _require(
        set(np.unique(labels["proxy"]).tolist()).issubset({2, 3}),
        "hairpin-support proxy contains an ordinary class",
    )
    return {
        **data,
        "labels": labels,
        "voxels": local_voxels,
        "resolution": local_resolution,
        "domain_min": domain_min,
        "domain_max": domain_max,
        "candidate_indices": data["candidate_indices"][member],
        "vortex_ids": data["vortex_ids"][member],
        "unique_voxel_count": int(len(np.unique(
            (local_voxels[:, 2] * local_resolution[1] + local_voxels[:, 1])
            * local_resolution[0]
            + local_voxels[:, 0]
        ))),
        "selection_record": {
            "rule": "target_vortex_ids > 0 over all valid ordered target rows",
            "prediction_independent": True,
            "selected_row_count": int(np.count_nonzero(member)),
            "selected_vortex_id_count": int(len(np.unique(data["vortex_ids"][member]))),
            "source_order_preserved": True,
            "global_bbox_derivation": (
                "componentwise minimum and maximum voxel indices over every "
                "selected true-hairpin-support row; no VortexId selection"
            ),
            "global_voxel_index_min_xyz_inclusive": minimum.tolist(),
            "global_voxel_index_max_xyz_inclusive": maximum.tolist(),
            "cropped_resolution_xyz": local_resolution.tolist(),
            "cropped_domain_min_xyz": domain_min.tolist(),
            "cropped_domain_max_xyz": domain_max.tolist(),
            "selected_candidate_index_sha256": _array_sha256(
                data["candidate_indices"][member]
            ),
        },
    }


def _axis_extent(data: dict[str, Any], axis: int) -> tuple[float, float]:
    return float(data["domain_min"][axis]), float(data["domain_max"][axis])


def _panel_label(fig: plt.Figure, ax: Any, label: str) -> None:
    offset = ScaledTranslation(-6 / 72, 4 / 72, fig.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes + offset,
        ha="right",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
    )


def _slug(view: str) -> str:
    return ("plus_" if view.startswith("+") else "minus_") + view[-1]


def _plot_view(
    data: dict[str, Any],
    projection: dict[str, Any],
    view: str,
    seed: int,
    figure_dir: Path,
    dpi: int,
    alignment_helper: Callable[..., dict[str, Any]],
    population_key: str,
    population_label: str,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    spec = VIEW_SPECS[view]
    figure_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = figure_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(METHODS), figsize=(13.0, 3.25))
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.20, top=0.76, wspace=0.12)
    colormap = ListedColormap(CLASS_COLORS)
    colormap.set_bad("#F2F2F2", alpha=1.0)
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), colormap.N)
    horizontal_extent = _axis_extent(data, spec.horizontal_axis)
    vertical_extent = _axis_extent(data, spec.vertical_axis)
    extent = (*horizontal_extent, *vertical_extent)
    panel_records: dict[str, Any] = {}
    for panel_index, (ax, method) in enumerate(zip(axes, METHODS)):
        labels = data["labels"][method]
        image = labels_to_projected_image(labels, projection)
        masked = np.ma.masked_less(image, 0)
        ax.imshow(
            masked,
            origin="lower",
            interpolation="nearest",
            cmap=colormap,
            norm=norm,
            extent=extent,
            aspect="auto",
            rasterized=True,
        )
        occupancy = image >= 0
        if np.any(occupancy) and np.any(~occupancy):
            ax.contour(
                occupancy.astype(np.float32),
                levels=(0.5,),
                colors=("#202020",),
                linewidths=(0.20,),
                origin="lower",
                extent=extent,
            )
        ax.set_xlim(horizontal_extent)
        ax.set_ylim(vertical_extent)
        ax.set_xlabel(AXIS_LABELS[spec.horizontal_axis])
        if panel_index == 0:
            ax.set_ylabel(AXIS_LABELS[spec.vertical_axis])
        else:
            ax.set_yticklabels([])
        ax.set_title(
            METHOD_LABELS[method],
            fontweight="bold" if method == "raw_fmt" else "normal",
        )
        ax.tick_params(length=2.0, width=0.6, pad=1.5)
        _panel_label(fig, ax, chr(ord("a") + panel_index))
        visible_rows = np.asarray(projection["visible_rows"], dtype=np.int64)
        panel_records[method] = {
            "input_class_counts": _class_counts(labels),
            "visible_class_counts": _class_counts(labels[visible_rows]),
            "input_row_count": int(len(labels)),
            "visible_row_count": int(len(visible_rows)),
            "occluded_row_count": int(len(labels) - len(visible_rows)),
            "excluded_row_count": 0,
        }

    title_population = (
        "four-class segmentation transfer"
        if population_key == "full_candidate"
        else "true-hairpin-support segmentation"
    )
    fig.suptitle(
        f"TBL {title_population} — {spec.title}, seed {seed}",
        y=0.985,
        fontsize=9.0,
    )
    fig.legend(
        handles=[
            Patch(facecolor=color, edgecolor="0.2", linewidth=0.25, label=label)
            for color, label in zip(CLASS_COLORS, CLASS_LABELS)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        columnspacing=1.3,
        handletextpad=0.45,
    )
    fig.text(
        0.5,
        0.035,
        (
            f"All {projection['input_row_count']:,} {population_label} cubes processed; "
            f"{projection['visible_row_count']:,} front cubes visible and "
            f"{projection['occluded_row_count']:,} occluded along {view}. "
            "Deterministic occupancy z-buffer; no sampling."
        ),
        ha="center",
        va="bottom",
        fontsize=6.0,
        color="0.25",
    )
    fig.canvas.draw()
    panel_ids = list("abcde")
    qa_stem = (
        _slug(view)
        if population_key == "full_candidate"
        else f"{population_key}_{_slug(view)}"
    )
    alignment_path = qa_dir / f"{qa_stem}_panel_alignment.json"
    alignment_overlay = qa_dir / f"{qa_stem}_panel_alignment_overlay.svg"
    alignment = alignment_helper(
        fig,
        axes=list(axes),
        panel_ids=panel_ids,
        row_groups=[panel_ids],
        json_out=alignment_path,
        overlay_svg=alignment_overlay,
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    output_population = (
        "" if population_key == "full_candidate" else f"_{population_key}"
    )
    prefix = figure_dir / (
        f"task4b_tbl{output_population}_segmentation_seed{seed}_{_slug(view)}"
    )
    paths = {
        "png": prefix.with_suffix(".png"),
        "pdf": prefix.with_suffix(".pdf"),
        "svg": prefix.with_suffix(".svg"),
    }
    fig.savefig(paths["png"], dpi=dpi)
    fig.savefig(paths["pdf"])
    fig.savefig(paths["svg"])
    plt.close(fig)
    return (
        {key: str(path.resolve()) for key, path in paths.items()},
        {
            "verdict": alignment.get("verdict"),
            "report": str(alignment_path.resolve()),
            "overlay": str(alignment_overlay.resolve()),
        },
        panel_records,
    )


def run_visualization(
    *,
    output_dir: Path,
    audit_path: Path,
    target_manifest_path: Path,
    figure_dir: Path,
    seed: int,
    views: tuple[str, ...],
    dpi: int,
    alignment_helper_dir: str | None = None,
) -> dict[str, Any]:
    _require(dpi >= 150, "DPI must be at least 150")
    _require(len(views) > 0 and len(set(views)) == len(views), "views must be unique")
    for view in views:
        _require(view in VIEW_SPECS, f"unknown view: {view}")
    data = _load_formal_inputs(
        output_dir.resolve(),
        audit_path.resolve(),
        target_manifest_path.resolve(),
        seed,
    )
    alignment_helper = _load_alignment_helper(alignment_helper_dir)
    population_inputs = {
        "full_candidate": {
            "data": data,
            "label": "full-candidate",
            "selection_record": {
                "rule": "all valid ordered target rows",
                "prediction_independent": True,
                "selected_row_count": int(len(data["voxels"])),
                "source_order_preserved": True,
                "selected_candidate_index_sha256": _array_sha256(
                    data["candidate_indices"]
                ),
            },
        },
        "true_hairpin_support_only": {
            "data": true_hairpin_support_subset(data),
            "label": "valid true-hairpin-support",
        },
    }
    population_inputs["true_hairpin_support_only"]["selection_record"] = (
        population_inputs["true_hairpin_support_only"]["data"]["selection_record"]
    )
    population_records: dict[str, Any] = {}
    for population_key, population_entry in population_inputs.items():
        population_data = population_entry["data"]
        view_records: dict[str, Any] = {}
        for view in views:
            projection = project_frontmost_rows(
                population_data["voxels"], population_data["resolution"], view
            )
            paths, alignment, panels = _plot_view(
                population_data,
                projection,
                view,
                seed,
                figure_dir,
                dpi,
                alignment_helper,
                population_key,
                population_entry["label"],
            )
            visible_rows = np.asarray(projection["visible_rows"], dtype=np.int64)
            view_records[view] = {
                "camera": VIEW_SPECS[view].title,
                "depth_selection": (
                    f"frontmost occupied cube = {VIEW_SPECS[view].front} "
                    f"index on axis {VIEW_SPECS[view].depth_axis}"
                ),
                "projected_image_shape": list(projection["image_shape"]),
                "input_row_count_before_projection": projection["input_row_count"],
                "visible_front_voxel_count_after_projection": projection[
                    "visible_row_count"
                ],
                "physically_occluded_voxel_count": projection["occluded_row_count"],
                "excluded_or_sampled_row_count": projection["excluded_row_count"],
                "all_input_rows_accounted_for": projection[
                    "all_input_rows_accounted_for"
                ],
                "common_visible_candidate_index_sha256": _array_sha256(
                    population_data["candidate_indices"][visible_rows]
                ),
                "same_visible_geometry_for_all_panels": True,
                "panels": panels,
                "outputs": paths,
                "output_sha256": {
                    kind: _sha256(Path(path)) for kind, path in paths.items()
                },
                "panel_alignment": alignment,
            }
        population_records[population_key] = {
            "selection": population_entry["selection_record"],
            "grid": {
                "resolution_xyz": population_data["resolution"].tolist(),
                "domain_min_xyz": population_data["domain_min"].tolist(),
                "domain_max_xyz": population_data["domain_max"].tolist(),
                "voxel_size_xyz": population_data["voxel_size"].tolist(),
            },
            "views": view_records,
        }

    payload = {
        "status": "PASS",
        "visualization_version": VISUALIZATION_VERSION,
        "experiment": EXPERIMENT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "figure_contract": {
            "core_conclusion": (
                "Compare one fixed-seed four-class TBL proxy segmentation with "
                "Raw, Raw-wide, FMT-only, and Raw+FMT on identical test cubes."
            ),
            "archetype": "image_plate_plus_data_integrity_validation",
            "backend": "python_matplotlib",
            "proxy_boundary": (
                "The target is the frozen physics proxy; TBL VortexIds provide "
                "manual hairpin support but not manual head/limb anatomy labels."
            ),
        },
        "seed_selection": {
            "seed": int(seed),
            "policy": "explicit CLI/default seed; no test-metric selection",
        },
        "ordered_test_rows": {
            "count": int(len(data["voxels"])),
            "sha256": data["ordered_row_sha256"],
            "unique_voxel_count": data["unique_voxel_count"],
            "all_methods_identical": True,
        },
        "grid": {
            "resolution_xyz": data["resolution"].tolist(),
            "domain_min_xyz": data["domain_min"].tolist(),
            "domain_max_xyz": data["domain_max"].tolist(),
            "voxel_size_xyz": data["voxel_size"].tolist(),
        },
        "taxonomy": [
            {"class_id": index, "name": name, "color": CLASS_COLORS[index]}
            for index, name in enumerate(CLASS_NAMES)
        ],
        "methods_in_panel_order": list(METHODS),
        "prediction_artifacts": data["prediction_records"],
        "audit": {
            "path": str(audit_path.resolve()),
            "sha256": data["audit_sha256"],
            "status": data["audit"].get("status"),
        },
        "target_manifest": {
            "path": str(target_manifest_path.resolve()),
            "sha256": data["target_manifest_sha256"],
        },
        "visualization_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "rendering": {
            "algorithm": "deterministic orthographic occupancy z-buffer",
            "sampling": False,
            "interior_visibility_rule": (
                "all rows enter each projection; only the frontmost occupied "
                "voxel per projected ray is visible, and every hidden row is "
                "counted as physically occluded"
            ),
            "image_interpolation": "nearest",
            "display_aspect": (
                "auto per panel for legibility; physical coordinate bounds and "
                "voxel-index occupancy are unchanged"
            ),
            "populations": population_records,
        },
    }
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = figure_dir / "visualization_manifest.json"
    payload["visualization_manifest"] = str(manifest_path.resolve())
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> dict[str, Any]:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    audit_path = Path(args.audit_json or output_dir / "independent_audit.json")
    target_manifest_path = Path(
        args.target_manifest
        or output_dir / "target_cache" / "target_cache_manifest.json"
    )
    figure_dir = Path(
        args.figure_dir
        or output_dir
        / "figures"
        / f"task4b_channel_to_tbl_2.3_seed{args.seed}"
    )
    return run_visualization(
        output_dir=output_dir,
        audit_path=audit_path,
        target_manifest_path=target_manifest_path,
        figure_dir=figure_dir,
        seed=args.seed,
        views=tuple(args.view or DEFAULT_VIEWS),
        dpi=args.dpi,
        alignment_helper_dir=args.alignment_helper_dir,
    )


if __name__ == "__main__":
    try:
        main()
    except (VisualizationInputError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"VISUALIZATION FAILED: {error}") from error
