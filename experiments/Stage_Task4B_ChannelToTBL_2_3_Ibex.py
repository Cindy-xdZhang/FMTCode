"""Make the frozen Task4-b 2.3 cache portable to an Ibex checkout.

The scientific config deliberately retains the user-provided Windows paths.
On POSIX those strings are relative paths, so the deployment copies the two
source VTK files under a literal ``C:/Users/...`` directory inside the staged
repository.  This script changes only manifest path spelling and the resulting
manifest hash; all input and chunk byte hashes must remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path("config/mainExp_Task4B_ChannelToTBL_2.3.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def stage(
    repo_root: str | Path = Path("."),
    config_path: str | Path = DEFAULT_CONFIG,
) -> Path:
    repo_root = Path(repo_root).resolve()
    config_path = _resolve(repo_root, config_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if spec.get("experiment") != "mainExp_Task4B_ChannelToTBL_2.3":
        raise RuntimeError("only the frozen Task4-b 2.3 config may be staged")
    output_dir = _resolve(repo_root, spec["output_dir"])
    snapshot_path = output_dir / "config_snapshot.yaml"
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    if json.dumps(snapshot, sort_keys=True) != json.dumps(spec, sort_keys=True):
        raise RuntimeError("config snapshot differs from the frozen config")

    manifest_path = output_dir / "target_cache" / "target_cache_manifest.json"
    manifest_before = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("target manifest config hash differs")
    mapping = {
        "target_flow": "flow_path",
        "target_gt": "gt_path",
        "target_vorticity_validation": "vorticity_validation_path",
    }
    staged_inputs = {}
    for manifest_key, config_key in mapping.items():
        configured_text = str(spec["target"][config_key])
        configured_path = _resolve(repo_root, configured_text)
        if not configured_path.is_file():
            raise FileNotFoundError(configured_path)
        actual_hash = _sha256(configured_path)
        if actual_hash != manifest[f"{manifest_key}_sha256"]:
            raise RuntimeError(f"staged {manifest_key} byte hash differs")
        manifest[manifest_key] = configured_text
        staged_inputs[manifest_key] = {
            "path_relative_to_repo": configured_path.relative_to(repo_root).as_posix(),
            "sha256": actual_hash,
        }
    manifest["config"] = config_path.relative_to(repo_root).as_posix()
    _atomic_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    manifest_after = _sha256(manifest_path)

    preparation_path = output_dir / "target_preparation_summary.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    preparation["target_cache_manifest"] = manifest_path.relative_to(repo_root).as_posix()
    preparation["target_cache_manifest_sha256"] = manifest_after
    _atomic_text(preparation_path, json.dumps(preparation, indent=2) + "\n")

    record_path = output_dir / "ibex_path_only_staging.json"
    record = {
        "experiment": spec["experiment"],
        "change_scope": "path_spelling_only_no_feature_or_label_bytes_changed",
        "manifest_sha256_before_path_rewrite": manifest_before,
        "manifest_sha256_after_path_rewrite": manifest_after,
        "config_sha256": _sha256(config_path),
        "staged_inputs": staged_inputs,
        "chunk_count": int(manifest["chunk_count"]),
        "chunk_hashes_preserved": True,
    }
    _atomic_text(record_path, json.dumps(record, indent=2) + "\n")

    fixed_files = {
        config_path,
        snapshot_path,
        manifest_path,
        preparation_path,
        record_path,
        _resolve(repo_root, spec["source"]["cache"]),
        repo_root / "ibex_bash" / "mainexp_task4b_channel_to_tbl_2p3_a100.sh",
    }
    for manifest_key in mapping:
        fixed_files.add(_resolve(repo_root, spec["target"][mapping[manifest_key]]))
    for directory in (repo_root / "FMT_Utils", repo_root / "DeepUtils", repo_root / "experiments"):
        fixed_files.update(
            path
            for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    fixed_files.update(
        path
        for path in (output_dir / "target_cache" / "chunks").glob("*.npz")
        if path.is_file()
    )
    lines = []
    for path in sorted(fixed_files, key=lambda value: value.relative_to(repo_root).as_posix()):
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(repo_root).as_posix()
        lines.append(f"{_sha256(path)}  {relative}")
    checksum_path = repo_root / "DEPLOYMENT_MANIFEST.sha256"
    _atomic_text(checksum_path, "\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": "READY",
                "experiment": spec["experiment"],
                "file_count": len(lines),
                "manifest_sha256": manifest_after,
                "deployment_checksum_sha256": _sha256(checksum_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return checksum_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    stage(args.repo_root, args.config)
