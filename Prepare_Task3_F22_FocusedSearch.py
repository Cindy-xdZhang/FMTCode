"""Freeze F22 stage-1 winners and build an isolated Task3 stage-2 config.

The source search has ten datasets, but F22 can proceed to stage 2 as soon as
its own development-only stage-1 rows are complete.  This helper never opens
outer-development or confirmation records.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from Search_Task3_FMTResidual_3D import (
    _candidate_summary,
    _load_spec,
    _selection_key,
    _write_csv,
)


DEFAULT_SOURCE = "config/Verify_Task3_FMTResidualFamilySearch_4.1.yaml"
DEFAULT_OUTPUT = "outputs/Verify_Task3_F22Hyperparams_1.1"
GROUP = "f22raptor"


def prepare(source_config=DEFAULT_SOURCE, output_root=DEFAULT_OUTPUT):
    source = _load_spec(source_config)
    if GROUP not in source["groups"]:
        raise ValueError(f"source config does not define group {GROUP!r}")
    if source["groups"][GROUP]["datasets"] != ["f22raptor"]:
        raise ValueError("focused F22 search requires a single-dataset group")

    rows = [
        _candidate_summary(source, GROUP, candidate)
        for candidate in source["candidates"]
    ]
    ranked = sorted(rows, key=_selection_key, reverse=True)
    top_k = int(source["selection"]["stage2_top_k"])
    selected = ranked[:top_k]
    for rank, row in enumerate(ranked, 1):
        row["rank_within_group"] = rank

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    leaderboard = output_root / "stage1_f22_leaderboard.csv"
    _write_csv(leaderboard, ranked)
    selection_path = output_root / "stage1_selection.json"
    selection = {
        "experiment": "Verify_Task3_F22Hyperparams_1.1",
        "source_experiment": source["experiment"],
        "selection_rule": (
            "F22 development-only: maximize validation F1 gain over the "
            "same-width Raw-PCA residual; tie-break by AP gain, worst seed, "
            "and absolute FMT F1"
        ),
        "opened_ordinals": sorted(
            set(source["screen_split"]["train_ordinals"])
            | set(source["screen_split"]["validation_ordinals"])
        ),
        "outer_ordinals_opened": False,
        "confirmation_opened": False,
        "top_k_by_group": {GROUP: selected},
        "primary_by_group": {GROUP: selected[0]},
    }
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )

    focused = copy.deepcopy(source)
    focused["experiment"] = "Verify_Task3_F22Hyperparams_1.1"
    focused["output_root"] = str(output_root)
    focused["datasets"] = ["f22raptor"]
    focused["groups"] = {GROUP: source["groups"][GROUP]}
    focused["selection"]["stage1_selection_file"] = str(selection_path)
    focused_config = output_root / "focused_config.yaml"
    focused_config.write_text(
        yaml.safe_dump(focused, sort_keys=False), encoding="utf-8"
    )
    audit = {
        "experiment": focused["experiment"],
        "source_config": str(source_config),
        "focused_config": str(focused_config),
        "stage1_selection": str(selection_path),
        "stage1_candidates_checked": len(rows),
        "retained_feature_ids": [row["candidate_id"] for row in selected],
        "outer_ordinals_opened": False,
        "confirmation_opened": False,
    }
    (output_root / "prepare_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return focused_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.source_config, args.output_root)
