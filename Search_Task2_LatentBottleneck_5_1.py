"""Development-only latent bottleneck search for 3D Task2.

Each physical family keeps the FMT block, hidden layers, optimizer settings,
and training budget selected by ``Verify_Task2_FMTVAEFamilySearch_4.1``.  The
only searched model hyperparameter is the latent dimension.  For every cell,
Raw and FMT use the same VAE settings and training seed.  Only development
ordinals 0--7 are opened; outer and confirmation populations remain closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from DeepUtils.utils import EasyConfig
from FMT_Utils.Task12Data_3D import load_cache_records, stack_reference
from Run_Task2_3D_Main import _prepare_inputs
from Search_Task2_FMTVAE_3D import _latent_metrics
from Verify_HighReVAE import _train


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_spec(path: str | Path) -> dict:
    path = Path(path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "experiment", "output_root", "source", "groups", "latent_dims",
        "splits", "screen_seeds", "kmeans_seed", "kmeans_n_init",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    if bool(spec.get("confirmation_opened", False)):
        raise ValueError("Task2 development search cannot open confirmation data")
    latent_dims = [int(value) for value in spec["latent_dims"]]
    if not latent_dims or min(latent_dims) < 1:
        raise ValueError("latent_dims must contain positive integers")
    if len(latent_dims) != len(set(latent_dims)):
        raise ValueError("latent_dims must be unique")
    train = {int(value) for value in spec["splits"]["selection_train"]}
    validation = {
        int(value) for value in spec["splits"]["selection_validation"]
    }
    forbidden = {int(value) for value in spec["splits"].get("forbidden", [])}
    if train & validation or (train | validation) & forbidden:
        raise ValueError("train, validation, and forbidden ordinals must be disjoint")
    datasets = []
    for name, group in spec["groups"].items():
        group_datasets = list(group["datasets"])
        if not group_datasets:
            raise ValueError(f"group {name!r} has no datasets")
        datasets.extend(group_datasets)
        settings = dict(group["base_vae"])
        required_settings = {
            "hidden_dims", "control_latent_dim", "beta", "learning_rate",
            "optimizer_steps", "relational_weight", "pair_count",
        }
        absent = sorted(required_settings.difference(settings))
        if absent:
            raise ValueError(f"group {name!r} missing base VAE keys: {absent}")
        if int(settings["control_latent_dim"]) not in latent_dims:
            raise ValueError(
                f"group {name!r} control latent dimension is absent from grid"
            )
        cache = str(group["development_cache"]).lower()
        if "confirmation" in cache or "test" in cache:
            raise ValueError(f"held-out cache forbidden in development search: {cache}")
    if len(datasets) != len(set(datasets)):
        raise ValueError("groups must partition unique datasets")
    spec["_config_path"] = str(path)
    spec["_datasets"] = datasets
    return spec


def _source_root(spec: dict) -> Path:
    source = spec["source"]
    env_name = str(source.get("repo_root_env", ""))
    value = os.environ.get(env_name) if env_name else None
    return Path(value or source.get("repo_root", ".")).resolve()


def _source_path(spec: dict, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _source_root(spec) / path


def _group_for_dataset(spec: dict, dataset: str) -> tuple[str, dict]:
    matches = [
        (name, group) for name, group in spec["groups"].items()
        if dataset in group["datasets"]
    ]
    if len(matches) != 1:
        raise ValueError(f"dataset {dataset!r} matched {len(matches)} groups")
    return matches[0]


def _candidate(spec: dict, group: dict, latent_index: int) -> dict:
    latent_dim = int(spec["latent_dims"][int(latent_index)])
    settings = dict(group["base_vae"])
    settings.pop("control_latent_dim")
    settings["hidden_dims"] = [int(value) for value in settings["hidden_dims"]]
    settings["latent_dim"] = latent_dim
    settings["id"] = f"latent_{latent_dim:03d}"
    return settings


def _decode_job(spec: dict, job_index: int) -> tuple[str, int]:
    datasets = list(spec["_datasets"])
    count = len(datasets) * len(spec["latent_dims"])
    index = int(job_index)
    if not 0 <= index < count:
        raise IndexError(f"job index {index} outside [0,{count})")
    dataset_index, latent_index = divmod(index, len(spec["latent_dims"]))
    return datasets[dataset_index], latent_index


def _load_development(spec: dict, dataset: str):
    _, group = _group_for_dataset(spec, dataset)
    ordinals = [
        *spec["splits"]["selection_train"],
        *spec["splits"]["selection_validation"],
    ]
    records = load_cache_records(
        _source_path(spec, group["development_cache"]),
        expected_count=int(spec.get("expected_slices", 10)),
        ordinals=ordinals,
    )
    by_ordinal = {int(record["ordinal"]): record for record in records}
    source = EasyConfig(str(_source_path(spec, group["source_config"])))
    train = [by_ordinal[int(value)] for value in spec["splits"]["selection_train"]]
    validation = [
        by_ordinal[int(value)]
        for value in spec["splits"]["selection_validation"]
    ]
    return train, validation, source


def _result_path(spec: dict, dataset: str, candidate: dict) -> Path:
    return (
        Path(spec["output_root"]) / "runs" / dataset
        / f"{candidate['id']}.csv"
    )


def run_candidate(config_path: str, job_index: int) -> Path:
    spec = _load_spec(config_path)
    dataset, latent_index = _decode_job(spec, job_index)
    group_name, group = _group_for_dataset(spec, dataset)
    candidate = _candidate(spec, group, latent_index)
    path = _result_path(spec, dataset, candidate)
    rows = _read_csv(path)
    expected_arms = {"raw", "fmt"}
    row_keys = [
        (row["arm"], int(row["training_seed"])) for row in rows
    ]
    if len(row_keys) != len(set(row_keys)):
        raise RuntimeError(f"duplicate paired rows in {path}")
    seed_arms = {}
    for row in rows:
        seed_arms.setdefault(int(row["training_seed"]), set()).add(row["arm"])
    incomplete = {
        seed: arms for seed, arms in seed_arms.items() if arms != expected_arms
    }
    if incomplete:
        raise RuntimeError(f"incomplete paired rows in {path}: {incomplete}")
    completed = {seed for seed, arms in seed_arms.items() if arms == expected_arms}

    train_records, validation_records, source = _load_development(spec, dataset)
    reference = stack_reference(validation_records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = {}
    for arm in ("raw", "fmt"):
        prepared[arm] = _prepare_inputs(
            train_records,
            validation_records,
            arm,
            str(group["fmt_feature"]),
            device,
        )

    for seed_value in spec["screen_seeds"]:
        seed = int(seed_value)
        if seed in completed:
            continue
        pair_rows = []
        pair_metrics = {}
        for arm in ("raw", "fmt"):
            train_x, validation_x = prepared[arm]
            train_mu, validation_mu, losses = _train(
                train_x, validation_x, candidate, source, seed, device
            )
            metrics = _latent_metrics(train_mu, validation_mu, reference, spec)
            pair_metrics[arm] = metrics
            pair_rows.append({
                "experiment": spec["experiment"],
                "group": group_name,
                "dataset": dataset,
                "arm": arm,
                "fmt_feature": str(group["fmt_feature"]),
                "source_architecture": str(group["source_architecture_id"]),
                "candidate": candidate["id"],
                "latent_dim": int(candidate["latent_dim"]),
                "control_latent_dim": int(
                    group["base_vae"]["control_latent_dim"]
                ),
                "training_seed": seed,
                "input_dim": int(train_x.shape[1]),
                "confirmation_opened": False,
                **metrics,
                **losses,
            })
        rows.extend(pair_rows)
        _write_csv(path, rows)
        completed.add(seed)
        print(
            f"{dataset}/{candidate['id']}/seed={seed}: "
            f"Raw={pair_metrics['raw']['f1']:.5f}, "
            f"FMT={pair_metrics['fmt']['f1']:.5f}, "
            f"gain={pair_metrics['fmt']['f1'] - pair_metrics['raw']['f1']:+.5f}",
            flush=True,
        )
    return path


def _selection_key(row: dict) -> tuple[float, float, float, float]:
    """Rank by the registered same-VAE gain objective."""
    return (
        float(row["fmt_minus_raw_f1_macro"]),
        float(row["worst_seed_f1_gain"]),
        float(row["worst_dataset_f1_gain"]),
        float(row["fmt_f1_macro"]),
    )


def _candidate_summary(spec: dict, group_name: str, latent_index: int) -> dict:
    group = spec["groups"][group_name]
    candidate = _candidate(spec, group, latent_index)
    seeds = [int(value) for value in spec["screen_seeds"]]
    per_dataset = {}
    seed_gains = {seed: [] for seed in seeds}
    for dataset in group["datasets"]:
        rows = _read_csv(_result_path(spec, dataset, candidate))
        indexed = {
            (row["arm"], int(row["training_seed"])): row for row in rows
        }
        expected = {(arm, seed) for arm in ("raw", "fmt") for seed in seeds}
        if len(rows) != len(expected) or set(indexed) != expected:
            raise RuntimeError(
                f"incomplete Task2 latent candidate {dataset}/{candidate['id']}: "
                f"expected {len(expected)} rows, found {len(indexed)}"
            )
        raw = np.asarray([
            float(indexed[("raw", seed)]["f1"]) for seed in seeds
        ])
        fmt = np.asarray([
            float(indexed[("fmt", seed)]["f1"]) for seed in seeds
        ])
        for index, seed in enumerate(seeds):
            seed_gains[seed].append(float(fmt[index] - raw[index]))
        per_dataset[dataset] = {
            "raw_f1": float(raw.mean()),
            "raw_f1_std": float(raw.std(ddof=0)),
            "fmt_f1": float(fmt.mean()),
            "fmt_f1_std": float(fmt.std(ddof=0)),
            "fmt_minus_raw_f1": float((fmt - raw).mean()),
            "fmt_minus_raw_f1_std": float((fmt - raw).std(ddof=0)),
        }
    raw_macro = float(np.mean([row["raw_f1"] for row in per_dataset.values()]))
    fmt_macro = float(np.mean([row["fmt_f1"] for row in per_dataset.values()]))
    seed_macro = {
        str(seed): float(np.mean(values)) for seed, values in seed_gains.items()
    }
    dataset_gains = [
        row["fmt_minus_raw_f1"] for row in per_dataset.values()
    ]
    return {
        "group": group_name,
        "candidate": candidate["id"],
        "latent_dim": int(candidate["latent_dim"]),
        "control_latent_dim": int(group["base_vae"]["control_latent_dim"]),
        "is_control": int(candidate["latent_dim"]) == int(
            group["base_vae"]["control_latent_dim"]
        ),
        "fmt_feature": str(group["fmt_feature"]),
        "source_architecture": str(group["source_architecture_id"]),
        "dataset_count": len(group["datasets"]),
        "raw_f1_macro": raw_macro,
        "fmt_f1_macro": fmt_macro,
        "fmt_minus_raw_f1_macro": fmt_macro - raw_macro,
        "worst_seed_f1_gain": min(seed_macro.values()),
        "worst_dataset_f1_gain": min(dataset_gains),
        "all_seed_gains_positive": min(seed_macro.values()) > 0.0,
        "all_dataset_gains_positive": min(dataset_gains) > 0.0,
        "seed_gains_json": json.dumps(seed_macro, sort_keys=True),
        "datasets_json": json.dumps(per_dataset, sort_keys=True),
    }


def select(config_path: str) -> Path:
    spec = _load_spec(config_path)
    leaderboard = []
    winners = {}
    controls = {}
    for group_name, group in spec["groups"].items():
        rows = [
            _candidate_summary(spec, group_name, index)
            for index in range(len(spec["latent_dims"]))
        ]
        ranked = sorted(rows, key=_selection_key, reverse=True)
        for rank, row in enumerate(ranked, 1):
            row["rank_within_group"] = rank
            leaderboard.append(row)
        winners[group_name] = ranked[0]
        matches = [row for row in rows if row["is_control"]]
        if len(matches) != 1:
            raise RuntimeError(f"group {group_name!r} has {len(matches)} controls")
        controls[group_name] = matches[0]

    def dataset_rows(selected: dict) -> list[dict]:
        result = []
        for group_name, row in selected.items():
            for dataset, metrics in json.loads(row["datasets_json"]).items():
                result.append({"group": group_name, "dataset": dataset, **metrics})
        return result

    winner_datasets = dataset_rows(winners)
    control_datasets = dataset_rows(controls)

    def aggregate(rows: list[dict]) -> dict:
        raw = float(np.mean([row["raw_f1"] for row in rows]))
        fmt = float(np.mean([row["fmt_f1"] for row in rows]))
        return {
            "raw_f1": raw,
            "fmt_f1": fmt,
            "fmt_minus_raw_f1": fmt - raw,
            "positive_datasets": sum(
                row["fmt_minus_raw_f1"] > 0.0 for row in rows
            ),
            "dataset_count": len(rows),
            "worst_dataset_f1_gain": min(
                row["fmt_minus_raw_f1"] for row in rows
            ),
        }

    winner_macro = aggregate(winner_datasets)
    control_macro = aggregate(control_datasets)
    family_gain = float(np.mean([
        row["fmt_minus_raw_f1_macro"] for row in winners.values()
    ]))
    output = Path(spec["output_root"])
    _write_csv(output / "leaderboard.csv", leaderboard)
    payload = {
        "experiment": spec["experiment"],
        "selection_rule": (
            "per physical family maximize paired same-VAE development F1 gain; "
            "tie-break by worst seed, worst dataset, then absolute FMT F1"
        ),
        "searched_factor": "latent_dim_only",
        "opened_ordinals": sorted(
            set(spec["splits"]["selection_train"])
            | set(spec["splits"]["selection_validation"])
        ),
        "forbidden_ordinals_opened": False,
        "confirmation_opened": False,
        "winner_by_group": winners,
        "control_by_group": controls,
        "winner_dataset_details": winner_datasets,
        "control_dataset_details": control_datasets,
        "winner_dataset_macro": winner_macro,
        "control_dataset_macro": control_macro,
        "winner_family_macro_f1_gain": family_gain,
        "gain_change_from_control": (
            winner_macro["fmt_minus_raw_f1"]
            - control_macro["fmt_minus_raw_f1"]
        ),
        "target_dataset_macro_f1_gain": float(
            spec.get("selection", {}).get(
                "target_dataset_macro_f1_gain", 0.22
            )
        ),
    }
    payload["target_reached"] = (
        winner_macro["fmt_minus_raw_f1"]
        >= payload["target_dataset_macro_f1_gain"]
    )
    payload["result_sha256"] = _json_sha256(payload)
    target = output / "selection.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(target.read_text(encoding="utf-8"))
    return target


def preflight(config_path: str) -> Path:
    spec = _load_spec(config_path)
    source_root = _source_root(spec)
    selection_path = _source_path(spec, spec["source"]["stage2_selection"])
    selection_sha = _sha256(selection_path)
    expected_sha = str(spec["source"]["stage2_selection_sha256"]).lower()
    if selection_sha != expected_sha:
        raise RuntimeError(
            f"stage2 selection SHA mismatch: {selection_sha} != {expected_sha}"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if bool(selection.get("confirmation_opened", False)):
        raise RuntimeError("source Task2 selection opened confirmation data")
    primary = selection["primary_by_group"]
    if set(primary) != set(spec["groups"]):
        raise RuntimeError("source selection and latent search groups differ")

    family_search_path = _source_path(
        spec, spec["source"]["family_search_config"]
    )
    family_search = yaml.safe_load(
        family_search_path.read_text(encoding="utf-8")
    )
    architecture_lookup = {
        str(row["id"]): dict(row)
        for row in family_search["stage2_architectures"]
    }
    cache_manifests = []
    for group_name, group in spec["groups"].items():
        selected = primary[group_name]
        if str(selected["fmt_feature"]) != str(group["fmt_feature"]):
            raise RuntimeError(f"FMT feature drift in group {group_name}")
        if str(selected["architecture"]) != str(group["source_architecture_id"]):
            raise RuntimeError(f"source architecture drift in group {group_name}")
        source_architecture = dict(
            architecture_lookup[str(group["source_architecture_id"])]
        )
        source_architecture.pop("id")
        frozen_base = dict(group["base_vae"])
        frozen_base["latent_dim"] = int(frozen_base.pop("control_latent_dim"))
        if source_architecture != frozen_base:
            raise RuntimeError(
                f"frozen VAE recipe drift in group {group_name}: "
                f"{source_architecture} != {frozen_base}"
            )
        for dataset in group["datasets"]:
            cache_dir = _source_path(spec, group["development_cache"]) / dataset
            manifest = cache_dir / "manifest.json"
            paths = sorted(cache_dir.glob("slice_*.npz"))
            if len(paths) != int(spec.get("expected_slices", 10)):
                raise RuntimeError(
                    f"expected {spec.get('expected_slices', 10)} slices in "
                    f"{cache_dir}, found {len(paths)}"
                )
            # Open only ordinals 0--7 to verify the exact development contract.
            records = load_cache_records(
                cache_dir,
                expected_count=int(spec.get("expected_slices", 10)),
                ordinals=[
                    *spec["splits"]["selection_train"],
                    *spec["splits"]["selection_validation"],
                ],
            )
            if any(not np.isfinite(record["raw"]).all() for record in records):
                raise ValueError(f"non-finite raw cache in {dataset}")
            if any(not np.isfinite(record["fmt"]).all() for record in records):
                raise ValueError(f"non-finite FMT cache in {dataset}")
            cache_manifests.append({
                "dataset": dataset,
                "manifest": str(manifest),
                "manifest_sha256": _sha256(manifest),
                "slice_count": len(paths),
                "opened_ordinals": [int(record["ordinal"]) for record in records],
            })

    mappings = [
        {"job_index": index, "dataset": _decode_job(spec, index)[0],
         "latent_dim": int(spec["latent_dims"][_decode_job(spec, index)[1]])}
        for index in range(len(spec["_datasets"]) * len(spec["latent_dims"]))
    ]
    config_path_obj = Path(config_path)
    payload = {
        "experiment": spec["experiment"],
        "config_path": str(config_path_obj),
        "config_sha256": _sha256(config_path_obj),
        "source_root": str(source_root),
        "source_selection": str(selection_path),
        "source_selection_sha256": selection_sha,
        "source_family_search_config": str(family_search_path),
        "source_family_search_config_sha256": _sha256(family_search_path),
        "source_confirmation_opened": False,
        "confirmation_opened": False,
        "opened_ordinals": sorted(
            set(spec["splits"]["selection_train"])
            | set(spec["splits"]["selection_validation"])
        ),
        "forbidden_ordinals": sorted(
            int(value) for value in spec["splits"].get("forbidden", [])
        ),
        "forbidden_ordinals_opened": False,
        "datasets": list(spec["_datasets"]),
        "latent_dims": [int(value) for value in spec["latent_dims"]],
        "screen_seeds": [int(value) for value in spec["screen_seeds"]],
        "candidate_mappings": mappings,
        "mapping_sha256": _json_sha256(mappings),
        "array_children": len(mappings),
        "paired_trainings": len(mappings) * len(spec["screen_seeds"]) * 2,
        "cache_manifests": cache_manifests,
        "checkpoint_policy": "no checkpoints are written",
    }
    payload["manifest_sha256"] = _json_sha256(payload)
    target = Path(spec["output_root"]) / "preflight_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(target.read_text(encoding="utf-8"))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode", choices=("preflight", "candidate", "select"), required=True
    )
    parser.add_argument("--job-index", type=int)
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight(args.config)
    elif args.mode == "select":
        select(args.config)
    else:
        if args.job_index is None:
            parser.error("candidate mode requires --job-index")
        run_candidate(args.config, args.job_index)


if __name__ == "__main__":
    main()
