"""Read-only formal figures for ``mainExp_Task4B_1.2``.

Figure contract
---------------
Core conclusion: determine whether Raw+FMT improves the held-out four-class
proxy-label macro-F1 over both Raw and the parameter-count control Raw-wide,
and show where the resulting classes lie in every held-out hairpin instance.
Results-level question: is any Raw+FMT gain consistent across the three frozen
optimizer seeds, classes, and all six fixed test VortexIds?
Archetypes: quantitative grid and image plate plus quantitative validation.
Quantitative panel a: four-variant macro-F1 mean +/- one sample standard
deviation with every seed visible.
Quantitative panel b: same-seed Raw+FMT minus Raw and Raw+FMT minus Raw-wide
macro-F1 differences.
Quantitative panel c: four-class F1 decomposition, again using mean +/- one
sample standard deviation.
Image plate: proxy, Raw, and Raw+FMT seed-consensus cube labels for the fixed
test VortexIds 38, 39, 51, 53, 62, and 76 under matched cameras and bounds.

The script consumes only audited formal artifacts. It never trains a model,
selects a seed or VortexId by test performance, or changes formal outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mandatory editable-text and journal typography settings. These are installed
# before any figure is created.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "DejaVu Sans",
    "Liberation Sans",
    "sans-serif",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.5,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5.5,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import ScaledTranslation
import numpy as np


DEFAULT_OUTPUT_DIR = Path("outputs/mainExp_Task4B_1.2/channel_GTs")
DEFAULT_DPI = 600
DEFAULT_CACHE = Path(
    "outputs/Other_Task4B_FMTFourClassClustering_1.2/channel_GTs/cache/"
    "task4b_four_class_cache.npz"
)

VARIANTS = ("raw", "raw_wide", "fmt_only", "raw_fmt")
VARIANT_LABELS = {
    "raw": "Raw",
    "raw_wide": "Raw-wide",
    "fmt_only": "FMT-only",
    "raw_fmt": "Raw+FMT",
}
VARIANT_COLORS = {
    "raw": "#606060",
    "raw_wide": "#8C8CC4",
    "fmt_only": "#4E9A75",
    "raw_fmt": "#C34F4A",
}
CLASS_NAMES = (
    "ordinary_streamwise",
    "ordinary_spanwise",
    "hairpin_head",
    "hairpin_limb",
)
CLASS_LABELS = {
    "ordinary_streamwise": "Ordinary streamwise",
    "ordinary_spanwise": "Ordinary spanwise",
    "hairpin_head": "Hairpin head",
    "hairpin_limb": "Hairpin limb",
}
CLASS_COLORS = {
    0: "#0072B2",
    1: "#E69F00",
    2: "#CC79A7",
    3: "#009E73",
}
TEST_VORTEX_IDS = (38, 39, 51, 53, 62, 76)
METHOD_ROWS = ("proxy", "raw", "raw_fmt")
METHOD_ROW_LABELS = {
    "proxy": "Proxy",
    "raw": "Raw consensus",
    "raw_fmt": "Raw+FMT consensus",
}
SEED_MARKERS = ("o", "s", "^")


class FigureInputError(RuntimeError):
    """Raised when formal figure inputs violate the frozen protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FigureInputError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument(
        "--audit-json",
        help="Independent audit JSON; defaults to OUTPUT_DIR/independent_audit.json",
    )
    parser.add_argument(
        "--figure-dir",
        help="Destination; defaults to OUTPUT_DIR/figures/task4b_supervised_1.2",
    )
    parser.add_argument(
        "--alignment-helper-dir",
        help=(
            "Directory containing audit_panel_alignment.py. If omitted, use "
            "NATURE_FIGURE_SCRIPTS or the installed nature-figure skill."
        ),
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
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
            Path(__file__).resolve().parents[1] / ".codex" / "nature-figure" / "scripts",
        ]
    )
    for directory in candidates:
        source = directory.resolve() / "audit_panel_alignment.py"
        if not source.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "task4b_audit_panel_alignment", source
        )
        _require(spec is not None and spec.loader is not None, f"cannot import {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        helper = getattr(module, "require_matplotlib_panel_alignment", None)
        _require(callable(helper), f"alignment helper is missing in {source}")
        return helper
    searched = ", ".join(str(path) for path in candidates)
    raise FigureInputError(
        "audit_panel_alignment.py was not found; provide "
        f"--alignment-helper-dir. Searched: {searched}"
    )


def _validate_distribution(
    distribution: dict[str, Any], expected_seeds: tuple[int, ...], context: str
) -> dict[int, float]:
    by_seed = {
        int(seed): float(value)
        for seed, value in distribution.get("per_seed", {}).items()
    }
    _require(tuple(sorted(by_seed)) == expected_seeds, f"{context}: seed set differs")
    values = np.asarray([by_seed[seed] for seed in expected_seeds], dtype=np.float64)
    _require(np.isfinite(values).all(), f"{context}: non-finite seed metric")
    mean = float(np.mean(values))
    sample_std = float(np.std(values, ddof=1))
    _require(
        np.isclose(mean, float(distribution["mean"]), rtol=0.0, atol=1e-12),
        f"{context}: reported mean differs from per-seed values",
    )
    _require(
        np.isclose(
            sample_std,
            float(distribution["sample_std"]),
            rtol=0.0,
            atol=1e-12,
        ),
        f"{context}: reported sample standard deviation differs",
    )
    return by_seed


def _load_audit(audit_path: Path, output_dir: Path, cache_path: Path) -> dict[str, Any]:
    _require(audit_path.is_file(), f"independent audit JSON is missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _require(audit.get("status") == "PASS", "independent audit status is not PASS")
    evidence = audit.get("evidence", {})
    _require(evidence.get("formal_run_count") == 12, "audit does not certify 12 runs")
    _require(tuple(evidence.get("variants", ())) == VARIANTS, "audit variant order differs")
    seeds = tuple(sorted(int(seed) for seed in evidence.get("seeds", ())))
    _require(len(seeds) == 3 and len(set(seeds)) == 3, "audit must certify three seeds")
    _require(
        tuple(int(value) for value in evidence.get("test_hairpin_vortex_ids", ()))
        == TEST_VORTEX_IDS,
        "audit test VortexId set differs from the frozen six-instance plate",
    )
    _require(cache_path.is_file(), f"frozen cache is missing: {cache_path}")
    actual_cache_sha = _sha256(cache_path)
    _require(
        audit.get("inputs", {}).get("cache_sha256") == actual_cache_sha,
        "audit/cache SHA-256 mismatch",
    )
    _require(
        Path(output_dir).is_dir(), f"formal output directory is missing: {output_dir}"
    )
    audited_output_dir = Path(audit.get("inputs", {}).get("output_dir", ""))
    _require(
        audited_output_dir.expanduser().resolve() == output_dir.resolve(),
        "audit JSON certifies a different formal output directory",
    )
    _require(
        evidence.get("metric_source")
        == "independently recomputed from prediction NPZ files",
        "audit does not certify independent metric recomputation",
    )
    runs = audit.get("runs", ())
    _require(len(runs) == 12, "audit run evidence does not contain 12 entries")
    seen_runs: set[tuple[str, int]] = set()
    for run in runs:
        key = (str(run.get("variant")), int(run.get("seed")))
        _require(key not in seen_runs, f"duplicate audited run: {key}")
        seen_runs.add(key)
        prediction_path = output_dir / "predictions" / f"{key[0]}_seed{key[1]}.npz"
        _require(prediction_path.is_file(), f"audited prediction is missing: {prediction_path}")
        _require(
            _sha256(prediction_path) == run.get("prediction_sha256"),
            f"prediction changed after the independent audit: {prediction_path.name}",
        )
    expected_runs = {(variant, seed) for variant in VARIANTS for seed in seeds}
    _require(seen_runs == expected_runs, "audited run set differs from the frozen design")
    return audit


def _extract_quantitative_rows(
    audit: dict[str, Any], seeds: tuple[int, ...]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    formal = audit.get("formal_metrics", {})
    variants = formal.get("variants", {})
    _require(tuple(variants) == VARIANTS, "formal metric variant order differs")
    rows: list[dict[str, Any]] = []
    panel_a: dict[str, Any] = {}
    for variant in VARIANTS:
        dist = variants[variant]["test"]["macro_f1"]
        per_seed = _validate_distribution(dist, seeds, f"{variant}.macro_f1")
        panel_a[variant] = dist
        for seed in seeds:
            rows.append(
                {
                    "panel": "a",
                    "metric": "test_macro_f1",
                    "variant": variant,
                    "comparison": "",
                    "class_name": "",
                    "seed": seed,
                    "value": per_seed[seed],
                    "mean": float(dist["mean"]),
                    "sample_std": float(dist["sample_std"]),
                    "spread_definition": "sample standard deviation across 3 seeds",
                }
            )

    paired = {
        "raw_fmt_minus_raw": formal["paired_raw_fmt_minus_raw"]["macro_f1"],
        "raw_fmt_minus_raw_wide": formal["paired_additional_controls"][
            "raw_fmt_minus_raw_wide"
        ]["macro_f1"],
    }
    for comparison, dist in paired.items():
        per_seed = _validate_distribution(dist, seeds, f"{comparison}.macro_f1")
        for seed in seeds:
            rows.append(
                {
                    "panel": "b",
                    "metric": "paired_test_macro_f1_difference",
                    "variant": "",
                    "comparison": comparison,
                    "class_name": "",
                    "seed": seed,
                    "value": per_seed[seed],
                    "mean": float(dist["mean"]),
                    "sample_std": float(dist["sample_std"]),
                    "spread_definition": "sample standard deviation across 3 paired seeds",
                }
            )

    panel_c: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        panel_c[variant] = {}
        available_classes = variants[variant].get("per_class_test", {})
        _require(
            tuple(available_classes) == CLASS_NAMES,
            f"{variant}: formal per-class order differs",
        )
        for class_name in CLASS_NAMES:
            dist = available_classes[class_name]["f1"]
            per_seed = _validate_distribution(
                dist, seeds, f"{variant}.{class_name}.f1"
            )
            panel_c[variant][class_name] = dist
            for seed in seeds:
                rows.append(
                    {
                        "panel": "c",
                        "metric": "test_per_class_f1",
                        "variant": variant,
                        "comparison": "",
                        "class_name": class_name,
                        "seed": seed,
                        "value": per_seed[seed],
                        "mean": float(dist["mean"]),
                        "sample_std": float(dist["sample_std"]),
                        "spread_definition": "sample standard deviation across 3 seeds",
                    }
                )
    return rows, {"panel_a": panel_a, "panel_b": paired, "panel_c": panel_c}


def _read_prediction(path: Path) -> dict[str, np.ndarray]:
    _require(path.is_file(), f"prediction file is missing: {path}")
    required = {
        "validation_source_indices",
        "validation_targets",
        "validation_probabilities",
        "test_source_indices",
        "test_targets",
        "test_probabilities",
    }
    with np.load(path, allow_pickle=False) as source:
        _require(set(source.files) == required, f"unexpected prediction keys: {path}")
        arrays = {
            key: np.asarray(source[key]).copy()
            for key in ("test_source_indices", "test_targets", "test_probabilities")
        }
    probabilities = arrays["test_probabilities"]
    _require(
        probabilities.ndim == 2 and probabilities.shape[1] == len(CLASS_NAMES),
        f"wrong test probability shape: {path}",
    )
    _require(np.isfinite(probabilities).all(), f"non-finite probabilities: {path}")
    _require(
        np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5),
        f"probability rows do not sum to one: {path}",
    )
    return arrays


def _load_cube_data(
    output_dir: Path, cache_path: Path, seeds: tuple[int, ...]
) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as cache:
        required_cache = {
            "labels",
            "split_codes",
            "seeds_xyz",
            "vortex_ids",
            "voxel_indices_xyz",
            "resolution_xyz",
            "domain_min_xyz",
            "domain_max_xyz",
        }
        _require(
            required_cache.issubset(cache.files),
            f"cache lacks keys: {sorted(required_cache - set(cache.files))}",
        )
        cache_arrays = {key: np.asarray(cache[key]).copy() for key in required_cache}
    labels = np.asarray(cache_arrays["labels"], dtype=np.int64).reshape(-1)
    split_codes = np.asarray(cache_arrays["split_codes"], dtype=np.int64).reshape(-1)
    vortex_ids = np.asarray(cache_arrays["vortex_ids"], dtype=np.int64).reshape(-1)
    indices = np.asarray(cache_arrays["voxel_indices_xyz"], dtype=np.int64)
    coordinates = np.asarray(cache_arrays["seeds_xyz"], dtype=np.float64)
    _require(indices.shape == (len(labels), 3), "cache voxel-index shape differs")
    _require(coordinates.shape == (len(labels), 3), "cache coordinate shape differs")
    _require(np.isin(labels, np.arange(4)).all(), "cache contains an unknown class")
    expected_test_source = np.flatnonzero(split_codes == 2).astype(np.int64)

    mean_probabilities: dict[str, np.ndarray] = {}
    probabilities_by_seed: dict[str, dict[int, np.ndarray]] = {}
    canonical_source: np.ndarray | None = None
    for variant in ("raw", "raw_fmt"):
        seed_probabilities: list[np.ndarray] = []
        probabilities_by_seed[variant] = {}
        for seed in seeds:
            arrays = _read_prediction(
                output_dir / "predictions" / f"{variant}_seed{seed}.npz"
            )
            source_indices = np.asarray(arrays["test_source_indices"], dtype=np.int64)
            targets = np.asarray(arrays["test_targets"], dtype=np.int64)
            _require(
                np.array_equal(source_indices, expected_test_source),
                f"{variant} seed {seed}: test source order differs from cache",
            )
            _require(
                np.array_equal(targets, labels[source_indices]),
                f"{variant} seed {seed}: targets differ from cache",
            )
            if canonical_source is None:
                canonical_source = source_indices
            else:
                _require(
                    np.array_equal(source_indices, canonical_source),
                    "prediction source orders differ",
                )
            probabilities = np.asarray(arrays["test_probabilities"], dtype=np.float64)
            seed_probabilities.append(probabilities)
            probabilities_by_seed[variant][seed] = probabilities
        mean_probabilities[variant] = np.mean(
            np.stack(seed_probabilities, axis=0), axis=0
        )
    _require(canonical_source is not None, "no formal prediction arrays were loaded")
    test_ids = tuple(
        int(value)
        for value in np.unique(vortex_ids[canonical_source][vortex_ids[canonical_source] > 0])
    )
    _require(test_ids == TEST_VORTEX_IDS, "prediction/cache test VortexIds differ")
    consensus = {
        variant: np.argmax(mean_probabilities[variant], axis=1).astype(np.int8)
        for variant in ("raw", "raw_fmt")
    }
    labels_by_seed = {
        variant: {
            seed: np.argmax(probabilities_by_seed[variant][seed], axis=1).astype(
                np.int8
            )
            for seed in seeds
        }
        for variant in ("raw", "raw_fmt")
    }
    return {
        "source_indices": canonical_source,
        "proxy_labels": labels[canonical_source].astype(np.int8),
        "vortex_ids": vortex_ids[canonical_source],
        "voxel_indices_xyz": indices[canonical_source],
        "coordinates_xyz": coordinates[canonical_source],
        "mean_probabilities": mean_probabilities,
        "probabilities_by_seed": probabilities_by_seed,
        "consensus_labels": consensus,
        "labels_by_seed": labels_by_seed,
        "resolution_xyz": np.asarray(cache_arrays["resolution_xyz"], dtype=np.int64),
        "domain_min_xyz": np.asarray(cache_arrays["domain_min_xyz"], dtype=np.float64),
        "domain_max_xyz": np.asarray(cache_arrays["domain_max_xyz"], dtype=np.float64),
    }


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _write_cube_source_csv(path: Path, cube: dict[str, Any]) -> int:
    rows: list[dict[str, Any]] = []
    source = cube["source_indices"]
    for position, source_index in enumerate(source):
        vortex_id = int(cube["vortex_ids"][position])
        if vortex_id not in TEST_VORTEX_IDS:
            continue
        proxy = int(cube["proxy_labels"][position])
        raw = int(cube["consensus_labels"]["raw"][position])
        raw_fmt = int(cube["consensus_labels"]["raw_fmt"][position])
        ix, iy, iz = (int(value) for value in cube["voxel_indices_xyz"][position])
        x, y, z = (float(value) for value in cube["coordinates_xyz"][position])
        row: dict[str, Any] = {
            "vortex_id": vortex_id,
            "source_index": int(source_index),
            "ix": ix,
            "iy": iy,
            "iz": iz,
            "x": x,
            "y": y,
            "z": z,
            "proxy_class_id": proxy,
            "proxy_class": CLASS_NAMES[proxy],
            "raw_consensus_class_id": raw,
            "raw_consensus_class": CLASS_NAMES[raw],
            "raw_fmt_consensus_class_id": raw_fmt,
            "raw_fmt_consensus_class": CLASS_NAMES[raw_fmt],
        }
        for variant in ("raw", "raw_fmt"):
            for class_id, class_name in enumerate(CLASS_NAMES):
                row[f"{variant}_mean_probability_{class_name}"] = float(
                    cube["mean_probabilities"][variant][position, class_id]
                )
            for seed, seed_labels in cube["labels_by_seed"][variant].items():
                label = int(seed_labels[position])
                row[f"{variant}_seed{seed}_class_id"] = label
                row[f"{variant}_seed{seed}_class"] = CLASS_NAMES[label]
                for class_id, class_name in enumerate(CLASS_NAMES):
                    row[f"{variant}_seed{seed}_probability_{class_name}"] = float(
                        cube["probabilities_by_seed"][variant][seed][
                            position, class_id
                        ]
                    )
        rows.append(row)
    fieldnames = list(rows[0]) if rows else []
    _require(bool(rows), "no fixed test VortexId rows were found")
    _write_csv(path, fieldnames, rows)
    return len(rows)


def _add_panel_label(ax: Any, label: str, fontsize: float = 8.0) -> None:
    offset = ScaledTranslation(-7 / 72, 4 / 72, ax.figure.dpi_scale_trans)
    text_method = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_method(
        0,
        1,
        label,
        transform=ax.transAxes + offset,
        fontsize=fontsize,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def _export_figure(fig: plt.Figure, prefix: Path, dpi: int) -> dict[str, str]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": prefix.with_suffix(".svg"),
        "pdf": prefix.with_suffix(".pdf"),
        "png": prefix.with_suffix(".png"),
        "tiff": prefix.with_suffix(".tiff"),
    }
    # Fixed canvas dimensions are intentional; bbox_inches='tight' is avoided so
    # the physical figure size and alignment audit cannot drift between formats.
    fig.savefig(paths["svg"])
    fig.savefig(paths["pdf"])
    fig.savefig(paths["png"], dpi=dpi)
    fig.savefig(
        paths["tiff"],
        dpi=dpi,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return {key: str(value.resolve()) for key, value in paths.items()}


def _plot_quantitative(
    metrics: dict[str, Any],
    seeds: tuple[int, ...],
    figure_dir: Path,
    dpi: int,
    require_matplotlib_panel_alignment: Callable[..., dict[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.23, top=0.77, wspace=0.34)

    ax = axes[0]
    x = np.arange(len(VARIANTS), dtype=float)
    means = np.asarray(
        [metrics["panel_a"][variant]["mean"] for variant in VARIANTS]
    )
    spreads = np.asarray(
        [metrics["panel_a"][variant]["sample_std"] for variant in VARIANTS]
    )
    ax.errorbar(
        x,
        means,
        yerr=spreads,
        fmt="none",
        ecolor="0.18",
        elinewidth=1.0,
        capsize=3.0,
        capthick=0.8,
        zorder=2,
    )
    for index, variant in enumerate(VARIANTS):
        ax.scatter(
            [index],
            [means[index]],
            s=45,
            color=VARIANT_COLORS[variant],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        per_seed = metrics["panel_a"][variant]["per_seed"]
        offsets = np.linspace(-0.10, 0.10, len(seeds))
        for offset, marker, seed in zip(offsets, SEED_MARKERS, seeds):
            ax.scatter(
                index + offset,
                float(per_seed[str(seed)]),
                s=16,
                marker=marker,
                facecolor="white",
                edgecolor=VARIANT_COLORS[variant],
                linewidth=0.8,
                zorder=4,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [VARIANT_LABELS[variant] for variant in VARIANTS],
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Four-class proxy reproduction")
    ax.grid(axis="y", color="0.90", linewidth=0.55)

    ax = axes[1]
    comparisons = ("raw_fmt_minus_raw", "raw_fmt_minus_raw_wide")
    comparison_labels = ("Raw+FMT − Raw", "Raw+FMT − Raw-wide")
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.8, zorder=1)
    for index, comparison in enumerate(comparisons):
        dist = metrics["panel_b"][comparison]
        values = np.asarray([dist["per_seed"][str(seed)] for seed in seeds])
        ax.errorbar(
            [index],
            [float(dist["mean"])],
            yerr=[float(dist["sample_std"])],
            fmt="o",
            ms=6.0,
            color="#C34F4A",
            ecolor="0.18",
            elinewidth=1.0,
            capsize=3.0,
            capthick=0.8,
            zorder=3,
        )
        offsets = np.linspace(-0.08, 0.08, len(seeds))
        for offset, marker, value in zip(offsets, SEED_MARKERS, values):
            ax.scatter(
                index + offset,
                value,
                s=18,
                marker=marker,
                facecolor="white",
                edgecolor="#C34F4A",
                linewidth=0.8,
                zorder=4,
            )
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(
        comparison_labels,
        rotation=22,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Paired macro-F1 difference")
    ax.set_title("Same-seed representation tests")
    ax.grid(axis="y", color="0.90", linewidth=0.55)

    ax = axes[2]
    class_x = np.arange(len(CLASS_NAMES), dtype=float)
    width = 0.18
    for variant_index, variant in enumerate(VARIANTS):
        centers = np.asarray(
            [metrics["panel_c"][variant][name]["mean"] for name in CLASS_NAMES]
        )
        spread = np.asarray(
            [
                metrics["panel_c"][variant][name]["sample_std"]
                for name in CLASS_NAMES
            ]
        )
        positions = class_x + (variant_index - 1.5) * width
        ax.bar(
            positions,
            centers,
            width=width,
            yerr=spread,
            color=VARIANT_COLORS[variant],
            edgecolor="white",
            linewidth=0.45,
            error_kw={"elinewidth": 0.7, "capsize": 1.8, "capthick": 0.7},
            label=VARIANT_LABELS[variant],
            zorder=2,
        )
    ax.set_xticks(class_x)
    ax.set_xticklabels(
        [
            "Ordinary\nstreamwise",
            "Ordinary\nspanwise",
            "Hairpin\nhead",
            "Hairpin\nlimb",
        ],
        rotation=0,
        ha="center",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Test class F1")
    ax.set_title("Class-wise performance")
    ax.grid(axis="y", color="0.90", linewidth=0.55, zorder=0)
    method_handles = [
        Patch(facecolor=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant])
        for variant in VARIANTS
    ]

    seed_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="0.25",
            markersize=4.0,
            label=f"Seed {seed}",
        )
        for marker, seed in zip(SEED_MARKERS, seeds)
    ]
    fig.legend(
        handles=seed_handles,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.985),
        ncol=3,
        handletextpad=0.3,
        columnspacing=0.9,
    )
    fig.legend(
        handles=method_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.905),
        ncol=4,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    fig.text(
        0.985,
        0.975,
        "Center ± sample SD; open symbols are individual seeds",
        ha="right",
        va="top",
        fontsize=5.5,
        color="0.30",
    )
    for label, axis in zip("abc", axes):
        _add_panel_label(axis, label, fontsize=8.0)

    fig.canvas.draw()
    qa_dir = figure_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    alignment = require_matplotlib_panel_alignment(
        fig,
        axes=list(axes),
        panel_ids=list("abc"),
        row_groups=[list("abc")],
        json_out=qa_dir / "quantitative_panel_alignment.json",
        overlay_svg=qa_dir / "quantitative_panel_alignment_overlay.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    paths = _export_figure(
        fig, figure_dir / "task4b_supervised_quantitative", dpi=dpi
    )
    plt.close(fig)
    return paths, alignment


def _labels_for_method(
    cube: dict[str, Any], method: str, seed: int | None = None
) -> np.ndarray:
    if method == "proxy":
        return np.asarray(cube["proxy_labels"], dtype=np.int8)
    if seed is not None:
        _require(seed in cube["labels_by_seed"][method], f"unknown seed: {seed}")
        return np.asarray(cube["labels_by_seed"][method][seed], dtype=np.int8)
    return np.asarray(cube["consensus_labels"][method], dtype=np.int8)


def _vortex_geometry(cube: dict[str, Any], vortex_id: int) -> dict[str, Any]:
    member = np.asarray(cube["vortex_ids"]) == int(vortex_id)
    _require(np.any(member), f"VortexId {vortex_id} has no test rows")
    indices = np.asarray(cube["voxel_indices_xyz"][member], dtype=np.int64)
    minimum = indices.min(axis=0)
    maximum = indices.max(axis=0)
    shape = tuple(int(value) for value in (maximum - minimum + 1))
    spacing = (
        np.asarray(cube["domain_max_xyz"], dtype=np.float64)
        - np.asarray(cube["domain_min_xyz"], dtype=np.float64)
    ) / np.asarray(cube["resolution_xyz"], dtype=np.float64)
    lower = np.asarray(cube["domain_min_xyz"], dtype=np.float64) + minimum * spacing
    upper = np.asarray(cube["domain_min_xyz"], dtype=np.float64) + (maximum + 1) * spacing
    edges = [
        np.linspace(lower[axis], upper[axis], shape[axis] + 1)
        for axis in range(3)
    ]
    mesh = np.meshgrid(*edges, indexing="ij")
    return {
        "member": member,
        "indices": indices,
        "minimum_index": minimum,
        "shape": shape,
        "lower": lower,
        "upper": upper,
        "mesh": mesh,
    }


def _draw_voxels(
    ax: Any,
    cube: dict[str, Any],
    geometry: dict[str, Any],
    method: str,
    elev: float,
    azim: float,
    show_axes: bool,
    seed: int | None = None,
) -> None:
    labels = _labels_for_method(cube, method, seed=seed)[geometry["member"]]
    local_indices = geometry["indices"] - geometry["minimum_index"]
    filled = np.zeros(geometry["shape"], dtype=bool)
    local_labels = np.full(geometry["shape"], -1, dtype=np.int8)
    x_index, y_index, z_index = local_indices.T
    filled[x_index, y_index, z_index] = True
    local_labels[x_index, y_index, z_index] = labels
    facecolors = np.zeros(geometry["shape"] + (4,), dtype=np.float32)
    for class_id, color in CLASS_COLORS.items():
        facecolors[local_labels == class_id] = to_rgba(color, alpha=1.0)
    ax.voxels(
        *geometry["mesh"],
        filled,
        facecolors=facecolors,
        edgecolor=(0.12, 0.12, 0.12, 0.38),
        linewidth=0.12,
        shade=False,
    )
    lower = geometry["lower"]
    upper = geometry["upper"]
    extent = np.maximum(upper - lower, np.finfo(float).eps)
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(extent)
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    if show_axes:
        ax.set_xlabel("x", labelpad=-6)
        ax.set_ylabel("y", labelpad=-6)
        ax.set_zlabel("z", labelpad=-7)
        ax.tick_params(pad=-2, length=1.5, width=0.45)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((0.65, 0.65, 0.65, 0.35))
        if np.isclose(elev, 0.0) and np.isclose(azim, 90.0):
            # The viewing direction is parallel to +y, so displaying the
            # collapsed y ticks creates overlapping labels without conveying
            # additional spatial information. The title records the view.
            ax.set_yticks([])
            ax.set_ylabel("")
    else:
        ax.set_axis_off()


def _class_legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=CLASS_COLORS[index], label=CLASS_LABELS[name])
        for index, name in enumerate(CLASS_NAMES)
    ]


def _plot_all_vortex_ids(
    cube: dict[str, Any],
    figure_dir: Path,
    dpi: int,
    require_matplotlib_panel_alignment: Callable[..., dict[str, Any]],
    seed: int | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    fig = plt.figure(figsize=(12.0, 6.6))
    axes = np.empty((len(METHOD_ROWS), len(TEST_VORTEX_IDS)), dtype=object)
    grid = fig.add_gridspec(
        len(METHOD_ROWS),
        len(TEST_VORTEX_IDS),
        left=0.055,
        right=0.99,
        bottom=0.035,
        top=0.89,
        wspace=0.02,
        hspace=0.02,
    )
    geometries = {vid: _vortex_geometry(cube, vid) for vid in TEST_VORTEX_IDS}
    panel_ids: list[str] = []
    for row, method in enumerate(METHOD_ROWS):
        for column, vortex_id in enumerate(TEST_VORTEX_IDS):
            ax = fig.add_subplot(grid[row, column], projection="3d")
            axes[row, column] = ax
            _draw_voxels(
                ax,
                cube,
                geometries[vortex_id],
                method,
                elev=22.0,
                azim=-55.0,
                show_axes=False,
                seed=seed,
            )
            if row == 0:
                ax.set_title(f"VortexId {vortex_id}", pad=0.5, fontsize=7.0)
            panel_id = chr(ord("a") + row * len(TEST_VORTEX_IDS) + column)
            panel_ids.append(panel_id)
            _add_panel_label(ax, panel_id, fontsize=6.0)
    for row, method in enumerate(METHOD_ROWS):
        box = axes[row, 0].get_position()
        fig.text(
            0.012,
            0.5 * (box.y0 + box.y1),
            (
                METHOD_ROW_LABELS[method]
                if method == "proxy" or seed is None
                else f"{VARIANT_LABELS[method]} seed {seed}"
            ),
            rotation=90,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold" if method == "raw_fmt" else "normal",
        )
    fig.legend(
        handles=_class_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=4,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    plate_description = (
        "All six fixed test instances; seed consensus = argmax(mean probability)"
        if seed is None
        else f"All six fixed test instances; frozen optimizer seed {seed}"
    )
    fig.text(
        0.99,
        0.985,
        plate_description,
        ha="right",
        va="top",
        fontsize=5.5,
        color="0.30",
    )
    fig.canvas.draw()
    flat_axes = list(axes.ravel())
    row_groups = [
        panel_ids[row * 6 : (row + 1) * 6] for row in range(len(METHOD_ROWS))
    ]
    column_groups = [
        [panel_ids[row * 6 + column] for row in range(len(METHOD_ROWS))]
        for column in range(6)
    ]
    qa_dir = figure_dir / "qa"
    alignment = require_matplotlib_panel_alignment(
        fig,
        axes=flat_axes,
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        json_out=qa_dir
        / (
            "cube_all_ids_consensus_panel_alignment.json"
            if seed is None
            else f"cube_all_ids_seed{seed}_panel_alignment.json"
        ),
        overlay_svg=qa_dir
        / (
            "cube_all_ids_consensus_panel_alignment_overlay.svg"
            if seed is None
            else f"cube_all_ids_seed{seed}_panel_alignment_overlay.svg"
        ),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    prefix = (
        "task4b_cube_segmentation_all_test_ids_consensus"
        if seed is None
        else f"task4b_cube_segmentation_all_test_ids_seed{seed}"
    )
    paths = _export_figure(fig, figure_dir / prefix, dpi=dpi)
    plt.close(fig)
    return paths, alignment


def _plot_each_vortex_id(
    cube: dict[str, Any],
    figure_dir: Path,
    dpi: int,
    require_matplotlib_panel_alignment: Callable[..., dict[str, Any]],
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    paths_by_id: dict[str, dict[str, str]] = {}
    alignment_by_id: dict[str, Any] = {}
    views = ((22.0, -55.0, "Isometric"), (0.0, 90.0, "View along +y (x–z)"))
    qa_dir = figure_dir / "qa"
    for vortex_id in TEST_VORTEX_IDS:
        geometry = _vortex_geometry(cube, vortex_id)
        fig = plt.figure(figsize=(6.0, 6.8))
        axes = np.empty((3, 2), dtype=object)
        grid = fig.add_gridspec(
            3,
            2,
            left=0.09,
            right=0.985,
            bottom=0.045,
            top=0.88,
            wspace=0.04,
            hspace=0.04,
        )
        panel_ids = list("abcdef")
        for row, method in enumerate(METHOD_ROWS):
            for column, (elev, azim, title) in enumerate(views):
                ax = fig.add_subplot(grid[row, column], projection="3d")
                axes[row, column] = ax
                _draw_voxels(
                    ax,
                    cube,
                    geometry,
                    method,
                    elev=elev,
                    azim=azim,
                    show_axes=False,
                )
                if row == 0:
                    ax.set_title(title, pad=1.0, fontsize=7.0)
                panel_id = panel_ids[row * 2 + column]
                _add_panel_label(ax, panel_id, fontsize=7.0)
        for row, method in enumerate(METHOD_ROWS):
            box = axes[row, 0].get_position()
            fig.text(
                0.018,
                0.5 * (box.y0 + box.y1),
                METHOD_ROW_LABELS[method],
                rotation=90,
                rotation_mode="anchor",
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="bold" if method == "raw_fmt" else "normal",
            )
        fig.suptitle(f"Held-out hairpin VortexId {vortex_id}", y=0.982, fontsize=8.0)
        fig.legend(
            handles=_class_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.55, 0.947),
            ncol=2,
            columnspacing=1.0,
            handletextpad=0.4,
        )
        lower = geometry["lower"]
        upper = geometry["upper"]
        fig.text(
            0.985,
            0.012,
            (
                "Shared physical bounds: "
                f"x {lower[0]:.3f}–{upper[0]:.3f}, "
                f"y {lower[1]:.3f}–{upper[1]:.3f}, "
                f"z {lower[2]:.3f}–{upper[2]:.3f}; "
                "x streamwise, y spanwise, z vertical"
            ),
            ha="right",
            va="bottom",
            fontsize=5.5,
            color="0.30",
        )
        fig.canvas.draw()
        alignment = require_matplotlib_panel_alignment(
            fig,
            axes=list(axes.ravel()),
            panel_ids=panel_ids,
            row_groups=[["a", "b"], ["c", "d"], ["e", "f"]],
            column_groups=[["a", "c", "e"], ["b", "d", "f"]],
            json_out=qa_dir / f"cube_vortex_{vortex_id}_panel_alignment.json",
            overlay_svg=qa_dir
            / f"cube_vortex_{vortex_id}_panel_alignment_overlay.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
        )
        paths = _export_figure(
            fig, figure_dir / f"task4b_cube_vortex_{vortex_id}", dpi=dpi
        )
        plt.close(fig)
        paths_by_id[str(vortex_id)] = paths
        alignment_by_id[str(vortex_id)] = alignment
    return paths_by_id, alignment_by_id


def _write_qa_notes(
    path: Path,
    audit_path: Path,
    cache_path: Path,
    quantitative_source_path: Path,
    cube_source_path: Path,
    cube_row_count: int,
    seeds: tuple[int, ...],
    alignments: dict[str, Any],
) -> None:
    verdicts = {
        key: value.get("verdict", "UNKNOWN") if isinstance(value, dict) else "UNKNOWN"
        for key, value in alignments.items()
    }
    text = f"""# Task4-b 1.2 figure QA notes

## Figure contract

- Core question: does Raw+FMT improve held-out four-class proxy reproduction over Raw and Raw-wide, and where do predicted classes lie in every held-out hairpin?
- Quantitative archetype: quantitative grid. Panel a establishes aggregate performance; panel b tests the paired representation and capacity-control differences; panel c identifies class-specific limits.
- Spatial archetype: image plate plus quantitative validation. It shows all fixed test VortexIds, not a performance-selected example.
- Backend: Python/Matplotlib Agg only.
- Exports: fixed-size SVG, PDF, TIFF, and PNG; no `bbox_inches='tight'`.

## Data and statistics

- Independent audit: `{audit_path.resolve()}` (`PASS` required before rendering).
- Frozen cache: `{cache_path.resolve()}`; SHA-256 is checked against the audit.
- Quantitative source data: `{quantitative_source_path.resolve()}`.
- Cube source data: `{cube_source_path.resolve()}` ({cube_row_count} rows).
- Replicate unit: optimizer seed, seeds={list(seeds)}.
- Center: arithmetic mean across three seeds.
- Spread: sample standard deviation across three seeds (`ddof=1`).
- No hypothesis-test p-value is displayed; three optimizer seeds do not establish population-level significance.
- Train/validation/test policy: the independent audit must certify the frozen buffered spatial split and test-once protocol.

## Image integrity and selection boundary

- VortexIds are frozen to {list(TEST_VORTEX_IDS)} and all six are shown.
- Proxy, Raw, and Raw+FMT rows use the same cached voxel centers, physical bounds, colors, and camera for each VortexId.
- Raw and Raw+FMT spatial labels are `argmax(mean class probability across all three frozen seeds)`; no seed is selected by test performance.
- Separate overview plates also show each frozen seed ({list(seeds)}) for both Raw and Raw+FMT, so seed instability is visible rather than hidden by consensus.
- Cube colors are categorical only; no interpolation, smoothing, morphology, cropping of source rows, brightness adjustment, or local contrast adjustment is applied.
- Per-instance detailed plates add an isometric view and a +y view. The overview uses the same isometric camera for every method.
- Different VortexIds use their own full cached bounding boxes so small instances remain visible; bounds never change between method rows for the same instance.

## Render-time checks

- Minimum configured text size: 5.5 pt. The exported PDFs still require `audit_pdf_text.py --min-pt 5`.
- Panel alignment verdicts: `{json.dumps(verdicts, sort_keys=True)}`.
- The exported PDFs still require `audit_figure_collisions.py` and final-size visual inspection. This script does not claim those post-export checks passed.
- Source preflight should be run with the nature-figure `validate_figure.py` helper before delivery.
"""
    path.write_text(text, encoding="utf-8")


def main() -> dict[str, Any]:
    args = _parse_args()
    _require(args.dpi >= 300, "DPI must be at least 300")
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_path = Path(args.cache).expanduser().resolve()
    audit_path = Path(args.audit_json).expanduser().resolve() if args.audit_json else (
        output_dir / "independent_audit.json"
    )
    figure_dir = Path(args.figure_dir).expanduser().resolve() if args.figure_dir else (
        output_dir / "figures" / "task4b_supervised_1.2"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    audit = _load_audit(audit_path, output_dir, cache_path)
    seeds = tuple(sorted(int(seed) for seed in audit["evidence"]["seeds"]))
    quantitative_rows, metrics = _extract_quantitative_rows(audit, seeds)
    cube = _load_cube_data(output_dir, cache_path, seeds)

    quantitative_source_path = figure_dir / "task4b_quantitative_source_data.csv"
    _write_csv(
        quantitative_source_path,
        (
            "panel",
            "metric",
            "variant",
            "comparison",
            "class_name",
            "seed",
            "value",
            "mean",
            "sample_std",
            "spread_definition",
        ),
        quantitative_rows,
    )
    cube_source_path = figure_dir / "task4b_cube_source_data.csv"
    cube_row_count = _write_cube_source_csv(cube_source_path, cube)

    require_matplotlib_panel_alignment = _load_alignment_helper(
        args.alignment_helper_dir
    )
    quantitative_paths, quantitative_alignment = _plot_quantitative(
        metrics, seeds, figure_dir, args.dpi, require_matplotlib_panel_alignment
    )
    cube_overview_paths, cube_overview_alignment = _plot_all_vortex_ids(
        cube, figure_dir, args.dpi, require_matplotlib_panel_alignment
    )
    cube_seed_overview_paths: dict[str, dict[str, str]] = {}
    cube_seed_overview_alignments: dict[str, Any] = {}
    for seed in seeds:
        seed_paths, seed_alignment = _plot_all_vortex_ids(
            cube,
            figure_dir,
            args.dpi,
            require_matplotlib_panel_alignment,
            seed=seed,
        )
        cube_seed_overview_paths[str(seed)] = seed_paths
        cube_seed_overview_alignments[str(seed)] = seed_alignment
    cube_detail_paths, cube_detail_alignments = _plot_each_vortex_id(
        cube, figure_dir, args.dpi, require_matplotlib_panel_alignment
    )
    alignments = {
        "quantitative": quantitative_alignment,
        "cube_all_ids_consensus": cube_overview_alignment,
        **{
            f"cube_all_ids_seed{seed}": value
            for seed, value in cube_seed_overview_alignments.items()
        },
        **{
            f"cube_vortex_{vortex_id}": value
            for vortex_id, value in cube_detail_alignments.items()
        },
    }
    qa_notes_path = figure_dir / "figure_qa_notes.md"
    _write_qa_notes(
        qa_notes_path,
        audit_path,
        cache_path,
        quantitative_source_path,
        cube_source_path,
        cube_row_count,
        seeds,
        alignments,
    )
    payload = {
        "status": "PASS",
        "audit_json": str(audit_path),
        "cache_sha256": _sha256(cache_path),
        "seeds": list(seeds),
        "fixed_test_vortex_ids": list(TEST_VORTEX_IDS),
        "quantitative_figure": quantitative_paths,
        "cube_overview_figure": cube_overview_paths,
        "cube_seed_overview_figures": cube_seed_overview_paths,
        "cube_detail_figures": cube_detail_paths,
        "quantitative_source_data": str(quantitative_source_path.resolve()),
        "cube_source_data": str(cube_source_path.resolve()),
        "figure_qa_notes": str(qa_notes_path.resolve()),
        "panel_alignment_verdicts": {
            key: value.get("verdict") for key, value in alignments.items()
        },
        "selection_policy": (
            "all six frozen test VortexIds and all three seeds; no test-based selection"
        ),
    }
    summary_path = figure_dir / "visualization_summary.json"
    payload["visualization_summary"] = str(summary_path.resolve())
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    try:
        main()
    except (FigureInputError, KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"VISUALIZATION FAILED: {error}") from error
