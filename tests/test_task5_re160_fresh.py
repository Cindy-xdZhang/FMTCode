from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Verify_Task5_Re160FreshTimes import (
    rank_adaptive_candidates,
    summarize_fresh_rows,
)


def test_adaptive_ranking_prefers_robust_candidate():
    rows = []
    for ordinal in (3, 4, 5):
        for seed in (60, 61):
            rows.extend((
                {"candidate_id": "", "ordinal": ordinal, "seed": seed,
                 "method": "raw", "f1": 0.50, "average_precision": 0.50},
                {"candidate_id": "", "ordinal": ordinal, "seed": seed,
                 "method": "raw_wide", "f1": 0.52, "average_precision": 0.52},
            ))
            for candidate, fmt in (
                ("brittle", 0.82 if ordinal == 3 else 0.48),
                ("robust", 0.64),
            ):
                rows.extend((
                    {"candidate_id": candidate, "ordinal": ordinal, "seed": seed,
                     "method": "raw_pca_residual", "f1": 0.53,
                     "average_precision": 0.53},
                    {"candidate_id": candidate, "ordinal": ordinal, "seed": seed,
                     "method": "fmt_residual", "f1": fmt,
                     "average_precision": fmt},
                ))
    ranked = rank_adaptive_candidates(
        rows, ["brittle", "robust"], [3, 4, 5]
    )
    assert ranked[0]["candidate_id"] == "robust"
    assert ranked[0]["worst_adaptive_gain"] > 0.10
    assert ranked[1]["worst_adaptive_gain"] < 0.0


def test_fresh_gate_uses_paired_seeds_and_both_baselines():
    rows = []
    for seed in range(70, 75):
        for method, f1, ap in (
            ("raw", 0.68, 0.69),
            ("raw_wide", 0.70, 0.71),
            ("raw_pca_residual", 0.72, 0.73),
            ("fmt_residual", 0.78, 0.80),
        ):
            rows.append({
                "seed": seed, "method": method,
                "f1": f1, "average_precision": ap,
            })
    summary = summarize_fresh_rows(rows, list(range(70, 75)), {
        "minimum_matched_gain": 0.03,
        "minimum_strong_raw_gain": 0.0,
        "minimum_positive_seed_count": 4,
    })
    assert summary["primary_gate_pass"]
    assert summary["seed_robustness_gate_pass"]
    assert summary["gains"]["f1_vs_raw_pca"] > 0.05
    assert summary["positive_seed_counts"]["f1_vs_raw_pca"] == 5


def test_fresh_times_and_scale_tuples_are_disjoint():
    root = Path(__file__).resolve().parents[1]
    old = yaml.safe_load(
        (root / "config/mainExp_Task5_3D_1.1.yaml").read_text(encoding="utf-8")
    )
    fresh = yaml.safe_load(
        (root / "config/Verify_Task5_Re160FreshTimes_1.1_cache.yaml")
        .read_text(encoding="utf-8")
    )
    old_tuples = {
        (row["offset_grid_scale"], row["dt_scale"], row["integration_steps"])
        for table in old["scale_sets"].values() for row in table
    }
    fresh_rows = fresh["scale_sets"]["fresh_confirmation"]
    fresh_tuples = {
        (row["offset_grid_scale"], row["dt_scale"], row["integration_steps"])
        for row in fresh_rows
    }
    assert old_tuples.isdisjoint(fresh_tuples)
    assert max(row["dt_scale"] * row["integration_steps"] for row in fresh_rows) == 8
    # Ten loaded frames: old development ends at 84, fresh windows are
    # [85,95) and [96,106), and old confirmation starts at 106.
    assert fresh["phases"]["confirmation"]["time_indices_by_dataset"]["cylinder3d"] == [85, 96]

