"""Freeze the best guarded Task3 recipe including adaptive follow-ups.

This stage trains nothing and never reads confirmation data.  It reuses the
validated 49.1 selector over five completed development searches, then copies
the selected seed-40/41 results and checkpoints into its own immutable output
tree so source-search cleanup cannot invalidate final confirmation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil

import yaml

import Select_Task3_FinalPortfolio_49_1 as _base


EXPERIMENT = "Verify_Task3_AdaptivePortfolio_52.1"
EXPECTED_SOURCES = {
    "safe_factor",
    "head_alpha_clip",
    "full_stack",
    "focal_gamma_low",
    "dropout_high",
}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_spec(path: str | Path) -> dict:
    path = Path(path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "experiment",
        "task",
        "output_root",
        "confirmation_opened",
        "sources",
        "datasets",
        "source_paired_seeds",
        "frozen_confirmation_seeds",
        "selection",
    }
    missing = sorted(required.difference(spec))
    if missing:
        raise ValueError(f"missing Task3 52.1 config keys: {missing}")
    if spec["experiment"] != EXPERIMENT:
        raise ValueError("Task3 52.1 experiment identity changed")
    if spec["task"] != "Task3" or bool(spec["confirmation_opened"]):
        raise ValueError("Task3 52.1 must remain development-only")
    if set(spec["sources"]) != EXPECTED_SOURCES:
        raise ValueError("Task3 52.1 source portfolio changed")

    datasets = [str(value) for value in spec["datasets"]]
    if len(datasets) != 10 or len(set(datasets)) != 10:
        raise ValueError("Task3 52.1 requires ten unique datasets")
    if [int(value) for value in spec["source_paired_seeds"]] != [40, 41, 42]:
        raise ValueError("Task3 52.1 source seeds changed")
    if [int(value) for value in spec["frozen_confirmation_seeds"]] != [40, 41]:
        raise ValueError("Task3 52.1 confirmation seeds changed")

    selection = dict(spec["selection"])
    metrics = [
        str(selection["primary_metric"]),
        *[str(value) for value in selection["tie_breakers"]],
    ]
    if len(metrics) != len(set(metrics)):
        raise ValueError("Task3 52.1 selection metrics are duplicated")
    if not bool(selection["require_source_absolute_fmt_guard"]):
        raise ValueError("Task3 52.1 requires guarded source winners")

    spec["datasets"] = datasets
    spec["source_paired_seeds"] = [40, 41, 42]
    spec["frozen_confirmation_seeds"] = [40, 41]
    spec["selection_metrics"] = metrics
    spec["config_path"] = str(path)
    spec["config_sha256"] = _sha256(path)
    return spec


@contextmanager
def _configured_base():
    previous = _base._load_spec
    try:
        _base._load_spec = _load_spec
        yield
    finally:
        _base._load_spec = previous


def static_preflight(config_path: str | Path) -> dict:
    with _configured_base():
        report = _base.static_preflight(config_path)
    if report["source_count"] != 5 or report["training_runs"] != 0:
        raise RuntimeError("Task3 52.1 static preflight contract changed")
    return report


def _copy_verified(source: Path, target: Path, expected_sha256: str) -> None:
    if not source.is_file() or _sha256(source) != expected_sha256:
        raise RuntimeError(f"source artifact changed: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _sha256(target) != expected_sha256:
            raise RuntimeError(f"frozen artifact collision: {target}")
    else:
        shutil.copy2(source, target)
    if _sha256(target) != expected_sha256:
        raise RuntimeError(f"frozen artifact copy failed: {target}")


def _freeze_local_copies(selection_path: Path) -> Path:
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if payload.get("experiment") != EXPERIMENT:
        raise RuntimeError("adaptive portfolio identity changed")
    models = list(payload.get("models", []))
    if len(models) != 40:
        raise RuntimeError("Task3 52.1 must freeze exactly 40 models")

    frozen_root = selection_path.parent / "frozen_artifacts"
    for model in models:
        relative = (
            Path(str(model["dataset"]))
            / f"seed{int(model['seed'])}"
            / str(model["source"])
        )
        source_result = Path(model["result"])
        source_checkpoint = Path(model["checkpoint"])
        target_result = frozen_root / relative / "per_run.csv"
        target_checkpoint = frozen_root / relative / source_checkpoint.name
        _copy_verified(source_result, target_result, model["result_sha256"])
        _copy_verified(
            source_checkpoint,
            target_checkpoint,
            model["checkpoint_sha256"],
        )
        model["source_result"] = str(source_result)
        model["source_checkpoint"] = str(source_checkpoint)
        model["result"] = str(target_result.resolve())
        model["checkpoint"] = str(target_checkpoint.resolve())

    payload["models"] = models
    payload["selection_rule"] = (
        "per physical family, maximize guarded development paired F1 gain "
        "across preregistered 44.1, 45.1, 48.1, 50.1, and 51.1 winners; "
        "use the registered absolute-FMT guards and tie breakers"
    )
    payload["frozen_model_count"] = len(models)
    payload["frozen_artifact_file_count"] = 2 * len(models)
    payload["frozen_artifact_root"] = str(frozen_root.resolve())
    temporary = selection_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(selection_path)
    return selection_path


def select(config_path: str | Path) -> Path:
    with _configured_base():
        target = _base.select(config_path)
    return _freeze_local_copies(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode", choices=("static-preflight", "select"), required=True
    )
    args = parser.parse_args()
    if args.mode == "static-preflight":
        static_preflight(args.config)
    else:
        target = select(args.config)
        print(target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
