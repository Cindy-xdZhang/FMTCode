"""Sealed Task3 confirmation of the adaptive 52.1 portfolio.

The development-only 52.1 selection and its copied checkpoints are frozen by
hash before the sixth spatial primitive population is generated. This module
performs no training, feature selection, threshold selection, residual-scale
selection, or hyperparameter selection on confirmation data.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path

import Build_Task3_AdaptiveTuned_Confirmation_7_2 as spatial
import Confirm_Task3_FinalTuned_7_1 as _base


SOURCE_EXPERIMENT = "Verify_Task3_AdaptivePortfolio_52.1"


@contextmanager
def _configured_base():
    """Temporarily route the validated 7.1 confirmation through 7.2 space."""
    previous = _base.spatial
    try:
        _base.spatial = spatial
        yield
    finally:
        _base.spatial = previous


def _load_spec(config_path: str | Path) -> dict:
    with _configured_base():
        return _base._load_spec(config_path)


def _source_state(spec: dict) -> tuple:
    with _configured_base():
        return _base._source_state(spec)


def _collect_models(spec: dict, source_root: Path, source: dict,
                    selection: dict) -> list[dict]:
    with _configured_base():
        return _base._collect_models(spec, source_root, source, selection)


def static_preflight(config_path: str | Path) -> Path:
    with _configured_base():
        return _base.static_preflight(config_path)


def freeze(config_path: str | Path) -> Path:
    with _configured_base():
        return _base.freeze(config_path)


def _frozen_state(spec: dict):
    with _configured_base():
        return _base._frozen_state(spec)


def source_preflight(config_path: str | Path) -> Path:
    with _configured_base():
        return _base.source_preflight(config_path)


def build_cache(config_path: str | Path, job_index: int,
                overwrite: bool = False) -> Path:
    with _configured_base():
        return _base.build_cache(config_path, job_index, overwrite)


def build_labels(config_path: str | Path, group_index: int,
                 overwrite: bool = False) -> Path:
    with _configured_base():
        return _base.build_labels(config_path, group_index, overwrite)


def evaluation_preflight(config_path: str | Path) -> Path:
    with _configured_base():
        return _base.evaluation_preflight(config_path)


def run_dataset(config_path: str | Path, dataset: str) -> Path:
    with _configured_base():
        return _base.run_dataset(config_path, dataset)


def summarize(config_path: str | Path) -> Path:
    """Aggregate all shards and publish a correctly identified 52.1 result."""
    captured = io.StringIO()
    with _configured_base(), redirect_stdout(captured):
        target = _base.summarize(config_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("experiment") != spatial.EXPERIMENT:
        raise RuntimeError("Task3 7.2 summary experiment changed")
    if payload.get("source_search_experiment") != SOURCE_EXPERIMENT:
        raise RuntimeError("Task3 7.2 summary source portfolio changed")
    payload["comparison"] = (
        "frozen 52.1 adaptive-portfolio FMT residual minus its same-recipe, "
        "same-capacity train-only Raw-PCA residual"
    )
    payload["source_portfolio_experiment"] = SOURCE_EXPERIMENT
    payload["source_portfolio_artifacts_copied_before_source_cleanup"] = True
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, target)
    print(target.read_text(encoding="utf-8"))
    return target


def _decode_dataset(spec: dict, index: int) -> str:
    if not 0 <= int(index) < len(spec["datasets"]):
        raise IndexError("Task3 7.2 dataset job outside [0,10)")
    return spec["datasets"][int(index)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "static-preflight", "freeze", "source-preflight", "cache",
            "labels", "evaluation-preflight", "dataset", "summary",
        ),
        required=True,
    )
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode == "static-preflight":
        static_preflight(args.config)
    elif args.mode == "freeze":
        freeze(args.config)
    elif args.mode == "source-preflight":
        source_preflight(args.config)
    elif args.mode == "cache":
        if args.job_index is None:
            parser.error("cache mode requires --job-index")
        build_cache(args.config, args.job_index, args.overwrite)
    elif args.mode == "labels":
        if args.job_index is None:
            parser.error("labels mode requires --job-index")
        build_labels(args.config, args.job_index, args.overwrite)
    elif args.mode == "evaluation-preflight":
        evaluation_preflight(args.config)
    elif args.mode == "dataset":
        spec = _load_spec(args.config)
        dataset = args.dataset
        if dataset is None:
            if args.job_index is None:
                parser.error("dataset mode requires --dataset or --job-index")
            dataset = _decode_dataset(spec, args.job_index)
        run_dataset(args.config, dataset)
    else:
        summarize(args.config)


if __name__ == "__main__":
    main()
