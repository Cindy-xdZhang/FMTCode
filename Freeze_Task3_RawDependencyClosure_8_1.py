"""Freeze and verify Raw checkpoints referenced by Task3 8.1 residuals.

PyTorch residual checkpoints contain a repository-relative ``raw_checkpoint``
path.  The 54.1 portfolio froze each residual checkpoint but omitted those Raw
dependencies.  This operational repair copies the exact dependencies without
unpickling checkpoint code, records their SHA-256 hashes, and never reads a
confirmation metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickletools
from pathlib import Path, PurePosixPath
import shutil
import zipfile

import yaml


EXPECTED_EXPERIMENT = "mainExp_Task3_3D_8.1"
EXPECTED_MODEL_COUNT = 40
EXPECTED_DEPENDENCY_COUNT = 20
MANIFEST_NAME = "raw_dependency_closure.json"
DEPENDENCY_DIR_NAME = "frozen_raw_dependencies"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checkpoint_pickle(path: str | Path) -> bytes:
    """Return ``data.pkl`` from a PyTorch ZIP without unpickling it."""
    with zipfile.ZipFile(path, "r") as archive:
        names = [
            name for name in archive.namelist()
            if name == "data.pkl" or name.endswith("/data.pkl")
        ]
        if len(names) != 1:
            raise RuntimeError(
                f"expected one data.pkl in checkpoint {path}, found {names}"
            )
        return archive.read(names[0])


def _extract_checkpoint_string(path: str | Path, key: str) -> str:
    """Read a literal string field using pickle disassembly, not execution."""
    waiting_for_value = False
    for opcode, argument, _ in pickletools.genops(_checkpoint_pickle(path)):
        if isinstance(argument, str):
            if waiting_for_value:
                return argument
            if argument == key:
                waiting_for_value = True
    raise RuntimeError(f"checkpoint {path} has no literal string field {key!r}")


def _safe_relative_path(value: str) -> Path:
    normalized = str(value).replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or ":" in pure.parts[0]
    ):
        raise ValueError(f"unsafe Raw checkpoint path: {value}")
    return Path(*pure.parts)


def _find_source_dependency(
    recorded_path: Path,
    source_checkpoint: str | Path,
    frozen_checkpoint: str | Path,
) -> Path:
    candidates: set[Path] = set()
    for checkpoint in (Path(source_checkpoint), Path(frozen_checkpoint)):
        for parent in checkpoint.resolve().parents:
            candidate = parent / recorded_path
            if candidate.is_file():
                candidates.add(candidate.resolve())
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one source Raw dependency for "
            f"{recorded_path}, found {sorted(str(path) for path in candidates)}"
        )
    return next(iter(candidates))


def _load_inputs(config_path: str | Path) -> tuple[dict, Path, dict]:
    config_path = Path(config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if spec.get("experiment") != EXPECTED_EXPERIMENT:
        raise RuntimeError("Task3 Raw dependency closure experiment changed")
    recipe_path = Path(spec["recipe_manifest"])
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe.get("experiment") != EXPECTED_EXPERIMENT:
        raise RuntimeError("Task3 Raw dependency recipe identity changed")
    models = list(recipe.get("models", []))
    if len(models) != EXPECTED_MODEL_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_MODEL_COUNT} frozen residual models, "
            f"found {len(models)}"
        )
    return spec, recipe_path, recipe


def _assert_no_confirmation_metrics(output_root: Path) -> None:
    forbidden = [output_root / "per_run.csv", output_root / "summary.json"]
    forbidden.extend((output_root / "shards").glob("*.csv"))
    existing = [path for path in forbidden if path.is_file()]
    if existing:
        raise RuntimeError(
            "cannot repair dependency closure after confirmation metrics exist: "
            + ", ".join(str(path) for path in existing)
        )


def _copy_verified(source: Path, target: Path, digest: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or _sha256(target) != digest:
            raise RuntimeError(f"Raw dependency target collision: {target}")
    else:
        shutil.copy2(source, target)
    if _sha256(target) != digest:
        raise RuntimeError(f"Raw dependency copy failed: {target}")


def freeze(config_path: str | Path) -> Path:
    spec, recipe_path, recipe = _load_inputs(config_path)
    output_root = Path(spec["output_root"]).resolve()
    _assert_no_confirmation_metrics(output_root)
    dependency_root = (output_root / DEPENDENCY_DIR_NAME).resolve()

    by_target: dict[str, dict] = {}
    observed_models: set[str] = set()
    paired_dependencies: dict[tuple[str, int], set[str]] = {}
    paired_arms: dict[tuple[str, int], set[str]] = {}
    for model in recipe["models"]:
        checkpoint = Path(model["checkpoint"])
        checkpoint_digest = str(model["checkpoint_sha256"])
        if not checkpoint.is_file() or _sha256(checkpoint) != checkpoint_digest:
            raise RuntimeError(f"frozen residual checkpoint changed: {checkpoint}")
        if checkpoint_digest in observed_models:
            raise RuntimeError(f"duplicate frozen model hash: {checkpoint_digest}")
        observed_models.add(checkpoint_digest)

        recorded = _safe_relative_path(
            _extract_checkpoint_string(checkpoint, "raw_checkpoint")
        )
        source_checkpoint = model.get("source_checkpoint")
        if not source_checkpoint:
            raise RuntimeError(f"model misses source_checkpoint: {checkpoint}")
        source = _find_source_dependency(
            recorded, source_checkpoint, checkpoint
        )
        digest = _sha256(source)
        target = (dependency_root / recorded).resolve()
        try:
            target.relative_to(dependency_root)
        except ValueError as error:
            raise RuntimeError(f"dependency target escapes closure: {target}") from error
        _copy_verified(source, target, digest)

        relative_key = recorded.as_posix()
        entry = by_target.setdefault(relative_key, {
            "recorded_path": relative_key,
            "source_path": str(source),
            "frozen_path": str(target),
            "sha256": digest,
            "referenced_models": [],
        })
        if entry["sha256"] != digest or entry["source_path"] != str(source):
            raise RuntimeError(f"Raw dependency collision: {relative_key}")
        reference = {
            "dataset": str(model["dataset"]),
            "seed": int(model["seed"]),
            "source": str(model["source"]),
            "checkpoint_sha256": checkpoint_digest,
        }
        entry["referenced_models"].append(reference)
        pair = (reference["dataset"], reference["seed"])
        paired_dependencies.setdefault(pair, set()).add(digest)
        paired_arms.setdefault(pair, set()).add(reference["source"])

    expected_pairs = {
        (str(model["dataset"]), int(model["seed"]))
        for model in recipe["models"]
    }
    if len(expected_pairs) != EXPECTED_DEPENDENCY_COUNT:
        raise RuntimeError("Task3 dependency pair count changed")
    for pair in expected_pairs:
        if paired_arms.get(pair) != {"fmt", "raw_pca"}:
            raise RuntimeError(f"Task3 dependency arms differ for {pair}")
        if len(paired_dependencies.get(pair, set())) != 1:
            raise RuntimeError(f"Task3 paired arms use different Raw models: {pair}")
    if len(by_target) != EXPECTED_DEPENDENCY_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_DEPENDENCY_COUNT} Raw dependencies, "
            f"found {len(by_target)}"
        )

    entries = []
    for key in sorted(by_target):
        entry = by_target[key]
        entry["referenced_models"] = sorted(
            entry["referenced_models"],
            key=lambda row: (row["dataset"], row["seed"], row["source"]),
        )
        entries.append(entry)
    payload = {
        "schema": 1,
        "experiment": EXPECTED_EXPERIMENT,
        "repair_kind": "operational_checkpoint_dependency_closure",
        "scientific_configuration_changed": False,
        "training_runs": 0,
        "confirmation_metrics_read": False,
        "config_sha256": _sha256(config_path),
        "recipe_manifest_sha256": _sha256(recipe_path),
        "source_model_selection_sha256": recipe[
            "source_model_selection_sha256"
        ],
        "dependency_root": str(dependency_root),
        "frozen_model_count": len(observed_models),
        "raw_dependency_count": len(entries),
        "entries": entries,
    }
    target = output_root / MANIFEST_NAME
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("Task3 Raw dependency closure changed")
    else:
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    print(target)
    return target


def verify(config_path: str | Path) -> Path:
    spec, recipe_path, recipe = _load_inputs(config_path)
    target = Path(spec["output_root"]).resolve() / MANIFEST_NAME
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "experiment": EXPECTED_EXPERIMENT,
        "scientific_configuration_changed": False,
        "training_runs": 0,
        "confirmation_metrics_read": False,
        "config_sha256": _sha256(config_path),
        "recipe_manifest_sha256": _sha256(recipe_path),
        "source_model_selection_sha256": recipe[
            "source_model_selection_sha256"
        ],
        "frozen_model_count": EXPECTED_MODEL_COUNT,
        "raw_dependency_count": EXPECTED_DEPENDENCY_COUNT,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Task3 Raw dependency closure changed: {key}")
    dependency_root = Path(payload["dependency_root"]).resolve()
    entries = list(payload.get("entries", []))
    if len(entries) != EXPECTED_DEPENDENCY_COUNT:
        raise RuntimeError("Raw dependency closure entry count changed")
    references: set[str] = set()
    for entry in entries:
        path = Path(entry["frozen_path"]).resolve()
        expected_path = (
            dependency_root
            / _safe_relative_path(str(entry["recorded_path"]))
        ).resolve()
        if path != expected_path:
            raise RuntimeError(f"dependency path mapping changed: {path}")
        try:
            path.relative_to(dependency_root)
        except ValueError as error:
            raise RuntimeError(f"dependency escaped frozen root: {path}") from error
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"frozen Raw dependency changed: {path}")
        referenced_models = list(entry.get("referenced_models", []))
        if len(referenced_models) != 2:
            raise RuntimeError(
                f"Raw dependency must serve one paired comparison: {path}"
            )
        if {str(row["source"]) for row in referenced_models} != {
            "fmt", "raw_pca"
        }:
            raise RuntimeError(f"Raw dependency paired arms changed: {path}")
        for model in referenced_models:
            references.add(str(model["checkpoint_sha256"]))
    expected_references = {
        str(model["checkpoint_sha256"]) for model in recipe["models"]
    }
    if references != expected_references:
        raise RuntimeError("Raw dependency closure does not cover all models")
    print(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("freeze", "verify"), required=True)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.config)
    else:
        verify(args.config)


if __name__ == "__main__":
    main()
