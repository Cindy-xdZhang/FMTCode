"""Freeze one F22 Task3 candidate using development records only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from Search_Task3_FMTResidual_3D import (
    _candidate_summary,
    _load_spec,
    _read_csv,
    _result_path,
    _write_csv,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_gains(spec: dict, candidate: dict) -> list[dict]:
    rows = []
    for seed_value in spec["screen_seeds"]:
        seed = int(seed_value)
        by_source = {}
        for source in ("fmt", "raw_pca"):
            values = _read_csv(
                _result_path(spec, candidate, "f22raptor", seed, source)
            )
            if len(values) != 1:
                raise RuntimeError(
                    f"incomplete result {candidate['id']}/seed{seed}/{source}"
                )
            by_source[source] = values[0]
        rows.append({
            "seed": seed,
            "f1_gain": (
                float(by_source["fmt"]["validation_f1"])
                - float(by_source["raw_pca"]["validation_f1"])
            ),
            "average_precision_gain": (
                float(by_source["fmt"]["validation_average_precision"])
                - float(by_source["raw_pca"]["validation_average_precision"])
            ),
        })
    return rows


def _rank_row(spec: dict, candidate: dict) -> dict:
    summary = _candidate_summary(spec, "f22raptor", candidate)
    paired = _seed_gains(spec, candidate)
    all_gains = [
        value
        for row in paired
        for value in (row["f1_gain"], row["average_precision_gain"])
    ]
    summary.update({
        "paired_seed_gains_json": json.dumps(paired, sort_keys=True),
        "worst_seed_metric_gain": float(min(all_gains)),
        "mean_min_metric_gain": float(min(
            summary["fmt_minus_raw_pca_f1_macro"],
            summary["fmt_minus_raw_pca_ap_macro"],
        )),
        "all_seed_metrics_positive": bool(min(all_gains) > 0.0),
    })
    return summary


def _key(row: dict) -> tuple[float, float, float, float]:
    return (
        float(row["worst_seed_metric_gain"]),
        float(row["mean_min_metric_gain"]),
        float(row["fmt_minus_raw_pca_f1_macro"]),
        float(row["fmt_ap_macro"]),
    )


def select(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    spec = _load_spec(config_path)
    if spec["datasets"] != ["f22raptor"]:
        raise ValueError("focused selector requires only f22raptor")
    confirmation = Path(spec["confirmation_schedule"])
    if not confirmation.exists():
        raise FileNotFoundError(confirmation)
    output = Path(spec["output_root"])
    target = output / "anchored_selection.json"
    if target.exists():
        raise FileExistsError(
            f"selection is already frozen: {target}; use a new experiment version"
        )
    rows = sorted(
        [_rank_row(spec, candidate) for candidate in spec["candidates"]],
        key=_key,
        reverse=True,
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    _write_csv(output / "anchored_leaderboard.csv", rows)
    winner_id = str(rows[0]["candidate_id"])
    winner = next(
        dict(candidate) for candidate in spec["candidates"]
        if str(candidate["id"]) == winner_id
    )
    threshold = float(
        spec["selection"].get("target_worst_seed_metric_gain", 0.005)
    )
    payload = {
        "experiment": spec["experiment"],
        "selection_data": "development train/validation only",
        "opened_ordinals": sorted(
            set(spec["screen_split"]["train_ordinals"])
            | set(spec["screen_split"]["validation_ordinals"])
        ),
        "confirmation_opened": False,
        "selection_rule": (
            "maximize the worst paired FMT-minus-Raw-PCA F1/AP gain across "
            "five seeds; tie-break by mean minimum gain, F1 gain, then FMT AP"
        ),
        "search_config_sha256": _sha256(config_path),
        "confirmation_schedule_sha256": _sha256(confirmation),
        "selected_candidate": winner,
        "selected_summary": rows[0],
        "target_worst_seed_metric_gain": threshold,
        "development_target_reached": bool(
            float(rows[0]["worst_seed_metric_gain"]) >= threshold
        ),
        "top_three": rows[:3],
    }
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(target.read_text(encoding="utf-8"))
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    select(args.config)
