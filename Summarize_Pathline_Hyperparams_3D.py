"""Summarize the three-axis 3D Task2 pathline hyperparameter experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import yaml


AXES = (
    ("integration_step", ("dt_small", "baseline", "dt_large"), "dt / source-frame interval",
     "Fixed horizon = 12 source-frame intervals"),
    ("total_steps", ("steps_short", "baseline", "steps_long"), "integration steps",
     "dt = 0.25; horizon changes with steps"),
    ("post_samples", ("samples_16", "baseline", "samples_48"), "post-integration samples",
     "Same integration: dt = 0.25, steps = 48"),
)


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _physical_families(spec):
    return {name: tuple(members) for name, members in
            spec["comparison"]["physical_families"].items()}


def _family_macro(dataset_values, families):
    family_values = [np.mean([dataset_values[name] for name in members])
                     for members in families.values()]
    return float(np.mean(family_values))


def _load_runs(spec):
    root = Path(spec["output_dir"])
    variant_by_id = {item["id"]: item for item in spec["variants"]}
    seeds = [int(spec["screening_seed"]), *map(int, spec["confirmation_seeds"])]
    rows, missing = [], []
    for variant_id, variant in variant_by_id.items():
        for dataset in spec["datasets"]:
            direct_path = root / "results_common" / variant_id / dataset / "direct.json"
            if not direct_path.exists():
                missing.append(str(direct_path)); continue
            canonical_direct = json.loads(direct_path.read_text(encoding="utf-8"))
            for seed in seeds:
                filename = "screen.json" if seed == int(spec["screening_seed"]) else f"seed_{seed}.json"
                path = root / "results_common" / variant_id / dataset / filename
                if not path.exists():
                    missing.append(str(path)); continue
                result = json.loads(path.read_text(encoding="utf-8"))
                legacy_direct = float(result["fmt_direct_f1"])
                result["fmt_direct_legacy_drift"] = abs(
                    legacy_direct - float(canonical_direct["fmt_direct_f1"])
                )
                result["fmt_direct_f1"] = float(canonical_direct["fmt_direct_f1"])
                rows.append({"variant": variant_id, "dataset": dataset,
                             "dt_scale": float(variant["dt_scale"]),
                             "integration_steps": int(variant["integration_steps"]),
                             "sampled_steps": int(variant["sampled_steps"]),
                             "physical_horizon_frames": (float(variant["dt_scale"]) *
                                                         int(variant["integration_steps"])),
                             "training_seed": seed, **result})
    if missing:
        preview = "\n".join(missing[:8])
        raise RuntimeError(f"missing {len(missing)} result files; first paths:\n{preview}")
    return rows


def _load_costs(spec):
    root = Path(spec["output_dir"]) / "cache"
    rows = []
    for variant in spec["variants"]:
        for dataset in spec["datasets"]:
            path = root / variant["id"] / dataset / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            slices = manifest["slices"]
            rows.append({
                "variant": variant["id"], "dataset": dataset,
                "build_seconds": float(sum(item["elapsed_seconds"] for item in slices)),
                "mean_slice_seconds": float(np.mean([item["elapsed_seconds"] for item in slices])),
                "native_valid_fraction": float(np.mean([
                    item["valid_primitives"] / item["total_primitives"] for item in slices
                ])),
            })
    return rows


def _summaries(spec, runs, costs):
    variant_ids = [item["id"] for item in spec["variants"]]
    seeds = [int(spec["screening_seed"]), *map(int, spec["confirmation_seeds"])]
    families = _physical_families(spec)
    summaries = []
    for variant in variant_ids:
        selected = [row for row in runs if row["variant"] == variant]
        direct_by_dataset = {}
        vae_family_by_seed, vae_all_by_seed = [], []
        recon_by_seed = []
        for dataset in spec["datasets"]:
            values = [float(row["fmt_direct_f1"]) for row in selected if row["dataset"] == dataset]
            if np.ptp(values) > 1e-12:
                raise RuntimeError(f"canonical direct FMT changed for {variant}/{dataset}")
            direct_by_dataset[dataset] = values[0]
        for seed in seeds:
            seed_rows = [row for row in selected if int(row["training_seed"]) == seed]
            vae_map = {row["dataset"]: float(row["fmt_vae_f1"]) for row in seed_rows}
            vae_family_by_seed.append(_family_macro(vae_map, families))
            vae_all_by_seed.append(float(np.mean(list(vae_map.values()))))
            recon_by_seed.append(float(np.mean([float(row["reconstruction"]) for row in seed_rows])))
        cost_rows = [row for row in costs if row["variant"] == variant]
        run_times = [float(row["train_seconds"]) for row in selected if "train_seconds" in row]
        summaries.append({
            "variant": variant,
            "fmt_direct_physical_family_f1": _family_macro(direct_by_dataset, families),
            "fmt_direct_all_entry_f1": float(np.mean(list(direct_by_dataset.values()))),
            "fmt_direct_max_legacy_drift": float(max(
                float(row["fmt_direct_legacy_drift"]) for row in selected
            )),
            "fmt_vae_physical_family_mean_f1": float(np.mean(vae_family_by_seed)),
            "fmt_vae_physical_family_std_f1": float(np.std(vae_family_by_seed)),
            "fmt_vae_physical_family_min_f1": float(np.min(vae_family_by_seed)),
            "fmt_vae_physical_family_max_f1": float(np.max(vae_family_by_seed)),
            "fmt_vae_all_entry_mean_f1": float(np.mean(vae_all_by_seed)),
            "fmt_vae_all_entry_std_f1": float(np.std(vae_all_by_seed)),
            "reconstruction_mean": float(np.mean(recon_by_seed)),
            "reconstruction_std": float(np.std(recon_by_seed)),
            "build_seconds_all_entries": float(sum(row["build_seconds"] for row in cost_rows)),
            "native_valid_fraction_all_entries": float(np.mean([
                row["native_valid_fraction"] for row in cost_rows
            ])),
            "common_valid_fraction_all_entries": float(np.mean([
                float(row["common_valid_fraction"]) for row in selected
            ])),
            "vae_train_seconds_mean": float(np.mean(run_times)) if run_times else float("nan"),
        })
    baseline = next(row for row in summaries if row["variant"] == "baseline")
    for row in summaries:
        row["build_time_ratio_to_baseline"] = (
            row["build_seconds_all_entries"] / baseline["build_seconds_all_entries"])
    return summaries


def _dataset_summaries(spec, runs, costs):
    output = []
    for variant in (item["id"] for item in spec["variants"]):
        for dataset in spec["datasets"]:
            selected = [row for row in runs if row["variant"] == variant and
                        row["dataset"] == dataset]
            cost = next(row for row in costs if row["variant"] == variant and
                        row["dataset"] == dataset)
            vae = np.asarray([float(row["fmt_vae_f1"]) for row in selected])
            reconstruction = np.asarray([float(row["reconstruction"]) for row in selected])
            output.append({
                "variant": variant, "dataset": dataset,
                "fmt_direct_f1": float(selected[0]["fmt_direct_f1"]),
                "fmt_direct_max_legacy_drift": float(max(
                    float(row["fmt_direct_legacy_drift"]) for row in selected
                )),
                "fmt_vae_mean_f1": float(vae.mean()), "fmt_vae_std_f1": float(vae.std()),
                "fmt_vae_min_f1": float(vae.min()), "fmt_vae_max_f1": float(vae.max()),
                "reconstruction_mean": float(reconstruction.mean()),
                "reconstruction_std": float(reconstruction.std()),
                "common_valid_fraction": float(selected[0]["common_valid_fraction"]),
                **{key: cost[key] for key in
                   ("build_seconds", "mean_slice_seconds", "native_valid_fraction")},
            })
    return output


def _effects(spec, runs):
    seeds = [int(spec["screening_seed"]), *map(int, spec["confirmation_seeds"])]
    families = _physical_families(spec)
    output = []
    for axis_name, variants, _, _ in AXES:
        for variant in variants:
            if variant == "baseline":
                continue
            direct_deltas = {}
            for dataset in spec["datasets"]:
                current = next(row for row in runs if row["variant"] == variant and
                               row["dataset"] == dataset)
                baseline = next(row for row in runs if row["variant"] == "baseline" and
                                row["dataset"] == dataset)
                direct_deltas[dataset] = (float(current["fmt_direct_f1"]) -
                                          float(baseline["fmt_direct_f1"]))
            vae_seed_deltas = []
            for seed in seeds:
                dataset_deltas = {}
                for dataset in spec["datasets"]:
                    current = next(row for row in runs if row["variant"] == variant and
                                   row["dataset"] == dataset and int(row["training_seed"]) == seed)
                    baseline = next(row for row in runs if row["variant"] == "baseline" and
                                    row["dataset"] == dataset and int(row["training_seed"]) == seed)
                    dataset_deltas[dataset] = (float(current["fmt_vae_f1"]) -
                                               float(baseline["fmt_vae_f1"]))
                vae_seed_deltas.append(_family_macro(dataset_deltas, families))
            output.append({
                "axis": axis_name, "variant": variant,
                "direct_physical_family_delta_f1": _family_macro(direct_deltas, families),
                "vae_physical_family_delta_mean_f1": float(np.mean(vae_seed_deltas)),
                "vae_physical_family_delta_std_f1": float(np.std(vae_seed_deltas)),
                "vae_physical_family_delta_min_f1": float(np.min(vae_seed_deltas)),
                "vae_physical_family_delta_max_f1": float(np.max(vae_seed_deltas)),
            })
    return output


def _dataset_effects(spec, runs):
    seeds = [int(spec["screening_seed"]), *map(int, spec["confirmation_seeds"])]
    output = []
    for axis_name, variants, _, _ in AXES:
        for variant in variants:
            if variant == "baseline":
                continue
            for dataset in spec["datasets"]:
                current = [row for row in runs if row["variant"] == variant and
                           row["dataset"] == dataset]
                baseline = [row for row in runs if row["variant"] == "baseline" and
                            row["dataset"] == dataset]
                direct_delta = (float(current[0]["fmt_direct_f1"]) -
                                float(baseline[0]["fmt_direct_f1"]))
                vae_deltas = []
                for seed in seeds:
                    current_seed = next(row for row in current if int(row["training_seed"]) == seed)
                    baseline_seed = next(row for row in baseline if int(row["training_seed"]) == seed)
                    vae_deltas.append(float(current_seed["fmt_vae_f1"]) -
                                      float(baseline_seed["fmt_vae_f1"]))
                output.append({
                    "axis": axis_name, "variant": variant, "dataset": dataset,
                    "direct_delta_f1": direct_delta,
                    "vae_delta_mean_f1": float(np.mean(vae_deltas)),
                    "vae_delta_std_f1": float(np.std(vae_deltas)),
                    "vae_delta_min_f1": float(np.min(vae_deltas)),
                    "vae_delta_max_f1": float(np.max(vae_deltas)),
                })
    return output


def _plot_axes(spec, summaries, output_dir):
    variant_cfg = {item["id"]: item for item in spec["variants"]}
    summary = {item["variant"]: item for item in summaries}
    fields = ("dt_scale", "integration_steps", "sampled_steps")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (axis_name, variants, xlabel, subtitle), field in zip(axes, AXES, fields):
        x = np.arange(len(variants))
        direct = [summary[name]["fmt_direct_physical_family_f1"] for name in variants]
        vae = [summary[name]["fmt_vae_physical_family_mean_f1"] for name in variants]
        errors = [summary[name]["fmt_vae_physical_family_std_f1"] for name in variants]
        ax.plot(x, direct, "o--", color="#f8961e", label="FMT direct")
        ax.errorbar(x, vae, yerr=errors, fmt="o-", capsize=4, color="#277da1",
                    label="FMT + VAE (3 seeds)")
        ax.set_xticks(x, [str(variant_cfg[name][field]) for name in variants])
        ax.set_xlabel(xlabel); ax.set_title(f"{axis_name}\n{subtitle}", fontsize=10)
        ax.grid(alpha=0.25); ax.set_ylim(0.0, 0.75)
    axes[0].set_ylabel("Held-out F1 (physical-family macro mean)")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(output_dir / "pathline_hyperparameter_effects.png", dpi=240)
    plt.close(fig)


def _plot_per_flow(spec, runs, output_dir):
    variants = ("dt_small", "dt_large", "steps_short", "steps_long", "samples_16", "samples_48")
    datasets = list(spec["datasets"])
    direct, vae = np.zeros((len(datasets), len(variants))), np.zeros((len(datasets), len(variants)))
    for row_index, dataset in enumerate(datasets):
        baseline_rows = [row for row in runs if row["variant"] == "baseline" and
                         row["dataset"] == dataset]
        baseline_direct = float(baseline_rows[0]["fmt_direct_f1"])
        baseline_vae = float(np.mean([float(row["fmt_vae_f1"]) for row in baseline_rows]))
        for column, variant in enumerate(variants):
            selected = [row for row in runs if row["variant"] == variant and
                        row["dataset"] == dataset]
            direct[row_index, column] = float(selected[0]["fmt_direct_f1"]) - baseline_direct
            vae[row_index, column] = float(np.mean([
                float(row["fmt_vae_f1"]) for row in selected])) - baseline_vae
    bound = max(float(np.abs(direct).max()), float(np.abs(vae).max()), 1e-3)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for ax, values, title in zip(axes, (direct, vae),
                                 ("FMT direct: ΔF1 vs baseline",
                                  "FMT + VAE: mean ΔF1 vs baseline (3 seeds)")):
        image = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(np.arange(len(variants)), variants, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(datasets)), datasets); ax.set_title(title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:+.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(values[i, j]) > bound * 0.55 else "black")
    fig.colorbar(image, ax=axes, shrink=0.82, label="Δ held-out F1")
    fig.savefig(output_dir / "pathline_hyperparameter_per_flow.png", dpi=240)
    plt.close(fig)


def _plot_cost(summaries, output_dir):
    names = [row["variant"] for row in summaries]
    ratios = [row["build_time_ratio_to_baseline"] for row in summaries]
    valid = [row["native_valid_fraction_all_entries"] for row in summaries]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].bar(names, ratios, color="#577590")
    axes[0].axhline(1.0, color="black", lw=1, ls="--")
    axes[0].set_ylabel("Cache-build time / baseline"); axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(names, valid, color="#43aa8b")
    axes[1].set_ylabel("Native valid primitive fraction"); axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=25); axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(output_dir / "pathline_hyperparameter_cost.png", dpi=240)
    plt.close(fig)


def _plot_reconstruction(spec, summaries, output_dir):
    variant_cfg = {item["id"]: item for item in spec["variants"]}
    summary = {item["variant"]: item for item in summaries}
    fields = ("dt_scale", "integration_steps", "sampled_steps")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (axis_name, variants, xlabel, subtitle), field in zip(axes, AXES, fields):
        x = np.arange(len(variants))
        means = [summary[name]["reconstruction_mean"] for name in variants]
        errors = [summary[name]["reconstruction_std"] for name in variants]
        ax.errorbar(x, means, yerr=errors, fmt="o-", capsize=4, color="#7b2cbf")
        ax.set_xticks(x, [str(variant_cfg[name][field]) for name in variants])
        ax.set_xlabel(xlabel); ax.set_title(f"{axis_name}\n{subtitle}", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Held-out FMT reconstruction MSE")
    fig.tight_layout(); fig.savefig(output_dir / "pathline_hyperparameter_reconstruction.png", dpi=240)
    plt.close(fig)


def summarize(spec_path):
    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    output_dir = Path(spec["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(spec); costs = _load_costs(spec)
    summaries = _summaries(spec, runs, costs); effects = _effects(spec, runs)
    dataset_summaries = _dataset_summaries(spec, runs, costs)
    dataset_effects = _dataset_effects(spec, runs)
    _write_csv(output_dir / "all_runs_common.csv", runs)
    _write_csv(output_dir / "build_costs.csv", costs)
    _write_csv(output_dir / "variant_summary.csv", summaries)
    _write_csv(output_dir / "dataset_summary.csv", dataset_summaries)
    _write_csv(output_dir / "effects_vs_baseline.csv", effects)
    _write_csv(output_dir / "dataset_effects_vs_baseline.csv", dataset_effects)
    payload = {"experiment": spec["experiment"],
               "families": _physical_families(spec),
               "variant_summary": summaries, "effects_vs_baseline": effects,
               "dataset_summary": dataset_summaries,
               "dataset_effects_vs_baseline": dataset_effects}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _plot_axes(spec, summaries, output_dir)
    _plot_per_flow(spec, runs, output_dir)
    _plot_cost(summaries, output_dir)
    _plot_reconstruction(spec, summaries, output_dir)
    print(json.dumps({"variant_summary": summaries,
                      "effects_vs_baseline": effects}, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/Verify_PathlineHyperparams3D_1.1.yaml")
    args = parser.parse_args(); summarize(args.config)
