"""Stage sources, build caches, and preflight the fixed-pathline search."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

from DeepUtils.utils import EasyConfig
from experiments.Build_Task2_Universality_Cache import build_dataset
from experiments.Prepare_Task3_Confirmation_SourcePacks_12_2 import _write_pack
from experiments.Run_Task123_FixedPathline import load_common_records


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def _spec(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path)
    return yaml.safe_load(path.read_text(encoding="utf-8")), path


def stage_boeing(config_path: str | Path, output: str | Path,
                  overwrite: bool = False) -> Path:
    spec, config_path = _spec(config_path)
    settings = spec["staging"]
    entry = settings["datasets"]["boeing747"]
    output = Path(output)
    details = _write_pack(
        "boeing747", Path(entry["path"]), output,
        [int(value) for value in entry["time_indices"]],
        int(settings["frame_count"]), int(settings["max_spatial_dim"]),
        overwrite=overwrite,
    )
    manifest = {
        "experiment": spec["experiment"],
        "config_sha256": _sha256(config_path),
        "kind": "exact_prestrided_temporal_window_pack",
        "equivalence": (
            "velocity values are exact float32 outputs of the production "
            "strided NetCDF window loader; no interpolation or filtering"
        ),
        "boeing747": details,
    }
    target = Path(spec["staged_window_root"]) / "source_staging_manifest.json"
    _atomic_json(target, manifest)
    print(target, flush=True)
    return target


def _decode_cache_job(spec: dict, job_index: int) -> tuple[dict, str]:
    datasets = [str(value) for value in spec["cache_sources"]["staged_new2"]]
    count = len(spec["variants"]) * len(datasets)
    if not 0 <= int(job_index) < count:
        raise IndexError(f"cache job index {job_index} outside [0,{count})")
    variant_index, dataset_index = divmod(int(job_index), len(datasets))
    return dict(spec["variants"][variant_index]), datasets[dataset_index]


def _build_payload(spec: dict, variant: dict, dataset: str, source_path: Path,
                   effective_indices: list[int], original_indices: list[int] | None,
                   staging_manifest: Path | None) -> EasyConfig:
    sampling = spec["sampling"]
    payload = {
        "experiment": spec["experiment"],
        "datasets": [{"id": dataset, "path": str(source_path), "enabled": True}],
        "sampling": {
            "timeslices": int(spec.get("expected_slices", 10)),
            "begin_fraction": 0.20,
            "end_fraction": 0.90,
            "max_spatial_dim": int(spec["staging"]["max_spatial_dim"]),
            "seed_grid_shape": list(sampling["seed_grid_shape"]),
            "boundary_fraction": float(sampling["boundary_fraction"]),
            "seed_grid_phase": sampling.get("seed_grid_phase"),
            "fixed_time_indices_by_dataset": {dataset: effective_indices},
        },
        "pathlines": {
            "method": str(spec["pathlines"]["method"]),
            "chunk_size": int(spec["pathlines"]["chunk_size"]),
            "offset_grid_scale": float(sampling["offset_grid_scale"]),
            "offset_mode": str(sampling["offset_mode"]),
            "dt_scale": float(variant["dt_scale"]),
            "integration_steps": int(variant["integration_steps"]),
            "sampled_steps": int(variant["sampled_steps"]),
        },
        "encoder": dict(spec["encoder"]),
        "reference": dict(spec["reference"]),
        "output": {
            "cache_dir": str(Path(spec["cache_root"]) / str(variant["id"])),
            "result_dir": str(Path(spec["output_root"]) / "unused_cache_results"),
        },
    }
    if original_indices is not None:
        payload["sampling"].update({
            "original_fixed_time_indices_by_dataset": {
                dataset: original_indices
            },
            "source_staging_manifest": str(staging_manifest),
            "source_staging_manifest_sha256": _sha256(staging_manifest),
        })
    config = EasyConfig()
    config.update(payload)
    return config


def build_cache(config_path: str | Path, job_index: int,
                boeing_pack: str | Path, smoke_path: str | Path,
                staging_manifest: str | Path, overwrite: bool = False) -> Path:
    spec, _ = _spec(config_path)
    variant, dataset = _decode_cache_job(spec, job_index)
    original = [int(value) for value in
                spec["staging"]["datasets"][dataset]["time_indices"]]
    if dataset == "boeing747":
        source_path = Path(boeing_pack)
        frame_count = int(spec["staging"]["frame_count"])
        effective = [ordinal * frame_count for ordinal in range(len(original))]
        original_indices = original
        manifest = Path(staging_manifest)
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        details = json.loads(manifest.read_text(encoding="utf-8"))
        if details.get("config_sha256") != _sha256(config_path):
            raise RuntimeError("Boeing staging manifest used another config")
        if details["boeing747"]["sha256"] != _sha256(source_path):
            raise RuntimeError("Boeing source-pack hash differs from manifest")
    elif dataset == "smokeBuoyancy":
        source_path = Path(smoke_path)
        effective = original
        original_indices = None
        manifest = None
    else:
        raise ValueError(f"unsupported staged dataset {dataset}")
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    config = _build_payload(
        spec, variant, dataset, source_path, effective, original_indices, manifest
    )
    target = build_dataset(config, dataset, overwrite=overwrite)
    print(f"DONE cache {variant['id']}/{dataset}: {target}", flush=True)
    return target


def preflight(config_path: str | Path) -> Path:
    spec, config_path = _spec(config_path)
    opened = sorted(int(value) for value in spec["selection"]["opened_ordinals"])
    rows = []
    for variant in [str(row["id"]) for row in spec["variants"]]:
        for dataset in spec["datasets"]:
            records = load_common_records(spec, variant, dataset, opened)
            rows.append({
                "variant": variant,
                "dataset": dataset,
                "opened_ordinals": opened,
                "slice_count": len(records),
                "common_valid_primitives": [
                    int(row["metadata"]["common_valid_primitives"])
                    for row in records
                ],
                "common_valid_fraction": [
                    float(row["metadata"]["common_valid_fraction"])
                    for row in records
                ],
                "source_indices": [
                    int(row["metadata"]["original_source_start_index"])
                    for row in records
                ],
            })
    payload = {
        "experiment": spec["experiment"],
        "status": "PASS",
        "config_sha256": _sha256(config_path),
        "confirmation_files_enumerated_but_not_opened": True,
        "opened_ordinals": opened,
        "combination_count": len(rows),
        "rows": rows,
    }
    target = Path(spec["output_root"]) / "cache_preflight.json"
    _atomic_json(target, payload)
    print(f"PASS {len(rows)} fixed pathline dataset combinations", flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/Verify_Task123_FixedPathline_1.1.yaml"
    )
    parser.add_argument(
        "--mode", required=True,
        choices=("stage-boeing", "build-cache", "preflight"),
    )
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--boeing-pack")
    parser.add_argument("--smoke-path")
    parser.add_argument("--staging-manifest")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode == "stage-boeing":
        if not args.output:
            parser.error("--output is required for stage-boeing")
        stage_boeing(args.config, args.output, args.overwrite)
    elif args.mode == "build-cache":
        required = {
            "--job-index": args.job_index,
            "--boeing-pack": args.boeing_pack,
            "--smoke-path": args.smoke_path,
            "--staging-manifest": args.staging_manifest,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            parser.error(f"build-cache requires {', '.join(missing)}")
        build_cache(
            args.config, args.job_index, args.boeing_pack, args.smoke_path,
            args.staging_manifest, args.overwrite,
        )
    else:
        preflight(args.config)


if __name__ == "__main__":
    main()
