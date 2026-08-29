import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from Search_Task3_LossOptimization_7_1 import (
    _decode_job,
    _load_optimization_spec,
    _merge_candidate_overrides,
    _merge_combination_recipe,
    _optimization_candidate,
    _resolve_combination_candidates,
)


CONFIG = "config/Verify_Task3_ProjectionFeatureCombination_24.1.yaml"


class Task3ProjectionFeatureCombinationTests(unittest.TestCase):
    def test_config_declares_four_cells_and_closed_confirmation(self):
        spec = _load_optimization_spec(CONFIG)
        self.assertEqual(
            set(spec["combination_sources"]),
            {"ultranarrow", "projection", "feature"},
        )
        self.assertEqual(
            spec["combination_sources"]["feature"]["kind"],
            "stage1_feature",
        )
        self.assertEqual(len(spec["optimization_candidates"]), 4)
        self.assertEqual(spec["paired_seeds"], [40, 41, 42])
        self.assertEqual(_decode_job(spec, 39), ("smokeBuoyancy", 3))
        self.assertFalse(spec["optimization_selection"]["confirmation_opened"])

    def test_feature_recipe_merges_and_conflicts_are_rejected(self):
        rows = {
            "projection": {
                "optimization_id": "p04",
                "optimization_recipe_json": json.dumps({
                    "id": "p04",
                    "model": {
                        "auxiliary_dim": 3,
                        "auxiliary_projection": "linear_rmsnorm_gelu",
                    },
                }),
            },
            "feature": {
                "optimization_id": "d11",
                "optimization_recipe_json": json.dumps({
                    "id": "d11", "fmt_feature": "aivd2w8_core",
                }),
            },
        }
        merged = _merge_combination_recipe(
            "combo", ["projection", "feature"], rows
        )
        self.assertEqual(merged["fmt_feature"], "aivd2w8_core")
        self.assertEqual(merged["model"]["auxiliary_dim"], 3)
        with self.assertRaisesRegex(ValueError, "frozen fmt_feature"):
            _merge_candidate_overrides(merged, {
                "id": "bad", "sources": ["projection", "feature"],
                "fmt_feature": "fmt_all",
            })

    def test_stage1_selector_is_normalized_and_hash_bound(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            projection = root / "projection.json"
            feature = root / "feature.json"
            projection.write_text(json.dumps({
                "experiment": "projection-exp",
                "confirmation_opened": False,
                "primary_by_group": {"family": {
                    "optimization_id": "p01",
                    "optimization_recipe_json": json.dumps({
                        "id": "p01", "model": {"auxiliary_dim": 2},
                    }),
                }},
            }), encoding="utf-8")
            feature.write_text(json.dumps({
                "experiment": "feature-exp",
                "confirmation_opened": False,
                "primary_by_group": {"family": {
                    "candidate_id": "d08", "fmt_feature": "aivd1w3_core",
                }},
            }), encoding="utf-8")
            spec = {
                "groups": {"family": {"datasets": ["flow"]}},
                "combination_sources": {
                    "projection": {
                        "selection": str(projection),
                        "expected_experiment": "projection-exp",
                        "kind": "optimization",
                    },
                    "feature": {
                        "selection": str(feature),
                        "expected_experiment": "feature-exp",
                        "kind": "stage1_feature",
                    },
                },
                "optimization_candidates": [{
                    "id": "combo", "sources": ["projection", "feature"],
                }],
            }
            resolved, hashes = _resolve_combination_candidates(spec)
            recipe = resolved["family"][0]
            self.assertEqual(recipe["fmt_feature"], "aivd1w3_core")
            self.assertEqual(recipe["model"]["auxiliary_dim"], 2)
            self.assertEqual(recipe["source_optimization_ids"], {
                "projection": "p01", "feature": "d08",
            })
            self.assertEqual(set(hashes), {"projection", "feature"})

    def test_resolved_feature_replaces_only_base_feature(self):
        spec = _load_optimization_spec(CONFIG)
        group = next(iter(spec["groups"]))
        dataset = spec["groups"][group]["datasets"][0]
        manifest = {
            "base_candidate_by_group": {group: {
                "id": "base", "fmt_feature": "old_feature",
                "upstream_candidate_id": "upstream",
                "training": {"learning_rate": 0.001},
            }},
            "optimization_candidates_by_group": {group: [{
                "id": "combo", "fmt_feature": "new_feature",
                "model": {"auxiliary_dim": 3},
            }] * 4},
        }
        candidate = _optimization_candidate(spec, manifest, dataset, 3)
        self.assertEqual(candidate["fmt_feature"], "new_feature")
        self.assertEqual(candidate["auxiliary_dim"], 3)
        self.assertEqual(candidate["training"]["learning_rate"], 0.001)


if __name__ == "__main__":
    unittest.main()
