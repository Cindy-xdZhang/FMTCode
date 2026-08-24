import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_seed_times_are_new_and_valid():
    config = yaml.safe_load((
        ROOT / "config" / "Confirm_Task3UniversalityKinematic_2.1.yaml"
    ).read_text(encoding="utf-8"))
    old_root = ROOT / "outputs" / "Verify_Task2Universality_1.1" / "cache"
    schedules = config["sampling"]["fixed_time_indices_by_dataset"]
    assert config["sampling"]["timeslices"] == 4
    for dataset, indices in schedules.items():
        assert indices == sorted(set(indices))
        old = []
        for path in sorted((old_root / dataset).glob("slice_*.npz")):
            old.append(int(path.stem.rsplit("_", 1)[-1]))
        assert len(old) == 10
        assert not set(indices) & set(old)
        total = 159 if dataset == "channel" else {
            "cylinder3d": 151, "halfcylinderRe640": 76,
            "halfcylinderRe6400": 151, "tangaroa": 201,
            "deltaWing_resampled": 171, "deltaWing_LBM": 234,
            "f22raptor": 159,
        }[dataset]
        assert min(indices) >= int(np.ceil(0.20 * total))
        assert max(indices) < int(np.floor(0.90 * total))
        assert max(indices) + 13 < total


def test_main_experiment_times_are_unseen_by_both_prior_protocols():
    final_config = yaml.safe_load((
        ROOT / "config" / "mainExp_Task3Universality_2.1_cache.yaml"
    ).read_text(encoding="utf-8"))
    exploratory_config = yaml.safe_load((
        ROOT / "config" / "Confirm_Task3UniversalityKinematic_2.1.yaml"
    ).read_text(encoding="utf-8"))
    old_root = ROOT / "outputs" / "Verify_Task2Universality_1.1" / "cache"
    exploratory = exploratory_config["sampling"]["fixed_time_indices_by_dataset"]
    final = final_config["sampling"]["fixed_time_indices_by_dataset"]
    totals = {
        "cylinder3d": 151, "halfcylinderRe640": 76,
        "halfcylinderRe6400": 151, "tangaroa": 201,
        "deltaWing_resampled": 171, "deltaWing_LBM": 234,
        "f22raptor": 159, "channel": 159,
    }
    assert set(final) == set(exploratory)
    for dataset, indices in final.items():
        old = {
            int(path.stem.rsplit("_", 1)[-1])
            for path in (old_root / dataset).glob("slice_*.npz")
        }
        assert len(indices) == 4
        assert indices == sorted(set(indices))
        assert not set(indices) & old
        assert not set(indices) & set(exploratory[dataset])
        assert min(indices) >= int(np.ceil(0.20 * totals[dataset]))
        assert max(indices) < int(np.floor(0.90 * totals[dataset]))
        assert max(indices) + 13 < totals[dataset]


def test_main_experiment_2_2_times_are_new_and_integrable():
    names = [
        "Confirm_Task3UniversalityKinematic_2.1.yaml",
        "mainExp_Task3Universality_2.1_cache.yaml",
    ]
    prior = [yaml.safe_load((ROOT / "config" / name).read_text(
        encoding="utf-8"
    ))["sampling"]["fixed_time_indices_by_dataset"] for name in names]
    final = yaml.safe_load((
        ROOT / "config" / "mainExp_Task3Universality_2.2_cache.yaml"
    ).read_text(encoding="utf-8"))["sampling"]["fixed_time_indices_by_dataset"]
    old_root = ROOT / "outputs" / "Verify_Task2Universality_1.1" / "cache"
    totals = {
        "cylinder3d": 151, "halfcylinderRe640": 76,
        "halfcylinderRe6400": 151, "tangaroa": 201,
        "deltaWing_resampled": 171, "deltaWing_LBM": 234,
        "f22raptor": 159, "channel": 159,
    }
    for dataset, indices in final.items():
        seen = {
            int(path.stem.rsplit("_", 1)[-1])
            for path in (old_root / dataset).glob("slice_*.npz")
        }
        for schedule in prior:
            seen.update(schedule[dataset])
        assert len(indices) == 4
        assert indices == sorted(set(indices))
        assert not set(indices) & seen
        assert min(indices) >= int(np.ceil(0.20 * totals[dataset]))
        assert max(indices) < int(np.floor(0.90 * totals[dataset]))
        assert max(indices) + 13 < totals[dataset]


def test_main_experiment_3_1_uses_eight_new_start_times():
    prior_config_names = [
        "Confirm_Task3UniversalityKinematic_2.1.yaml",
        "mainExp_Task3Universality_2.1_cache.yaml",
        "mainExp_Task3Universality_2.2_cache.yaml",
    ]
    prior_schedules = [yaml.safe_load((ROOT / "config" / name).read_text(
        encoding="utf-8"
    ))["sampling"]["fixed_time_indices_by_dataset"] for name in prior_config_names]
    old_root = ROOT / "outputs" / "Verify_Task2Universality_1.1" / "cache"
    config = yaml.safe_load((
        ROOT / "config" / "mainExp_Task3_3D_3.1_confirmation_old8.yaml"
    ).read_text(encoding="utf-8"))
    totals = {
        "cylinder3d": 151, "halfcylinderRe640": 76,
        "halfcylinderRe6400": 151, "tangaroa": 201,
        "deltaWing_resampled": 171, "deltaWing_LBM": 234,
        "f22raptor": 159, "channel": 159,
    }
    assert config["sampling"]["timeslices"] == 8
    for dataset, indices in config["sampling"]["fixed_time_indices_by_dataset"].items():
        seen = {
            int(path.stem.rsplit("_", 1)[-1])
            for path in (old_root / dataset).glob("slice_*.npz")
        }
        for schedule in prior_schedules:
            seen.update(schedule[dataset])
        assert indices == sorted(set(indices))
        assert len(indices) == 8
        assert not set(indices) & seen
        assert min(indices) >= int(np.ceil(0.20 * (totals[dataset] - 1)))
        assert max(indices) + 13 < totals[dataset]

    new2 = yaml.safe_load((
        ROOT / "config" / "mainExp_Task3_3D_3.1_confirmation_new2.yaml"
    ).read_text(encoding="utf-8"))
    prior = yaml.safe_load((
        ROOT / "config" / "mainExp_Task123NewFlows_1.1_confirmation_cache.yaml"
    ).read_text(encoding="utf-8"))["sampling"]["fixed_time_indices_by_dataset"]
    development = yaml.safe_load((
        ROOT / "config" / "mainExp_Task123NewFlows_1.1_development_cache.yaml"
    ).read_text(encoding="utf-8"))["sampling"]["fixed_time_indices_by_dataset"]
    new_totals = {"boeing747": 199, "smokeBuoyancy": 160}
    for dataset, indices in new2["sampling"]["fixed_time_indices_by_dataset"].items():
        seen = set(prior[dataset]) | set(development[dataset])
        assert indices == sorted(set(indices))
        assert len(indices) == 8
        assert not set(indices) & seen
        assert min(indices) >= int(np.ceil(0.20 * (new_totals[dataset] - 1)))
        assert max(indices) + 13 < new_totals[dataset]


if __name__ == "__main__":
    test_confirmation_seed_times_are_new_and_valid()
    test_main_experiment_times_are_unseen_by_both_prior_protocols()
    test_main_experiment_2_2_times_are_new_and_integrable()
    test_main_experiment_3_1_uses_eight_new_start_times()
    print("TASK3 CONFIRMATION SCHEDULE TEST PASSED")
