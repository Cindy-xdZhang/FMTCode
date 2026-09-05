import numpy as np

from FMT_Utils.Task4B_ProxyLabels_3D import (
    HAIRPIN_HEAD,
    HAIRPIN_LIMB,
    ORDINARY_SPANWISE,
    ORDINARY_STREAMWISE,
    apply_cluster_mapping,
    best_cluster_permutation,
    build_four_vortex_type_labels,
    four_class_metrics,
    select_recall_constrained_ivd,
    split_vortex_ids_by_buffered_x,
    vorticity_deviation_volume,
)


def test_channel_profile_deviation_removes_z_dependent_mean():
    omega = np.zeros((3, 2, 2, 3), dtype=np.float32)
    omega[..., 1] = np.arange(3, dtype=np.float32)[:, None, None]
    standard, _ = vorticity_deviation_volume(omega, "whole_domain")
    profile, mean = vorticity_deviation_volume(omega, "wall_normal_plane")
    assert np.max(standard) > 0.0
    assert np.array_equal(profile, np.zeros_like(profile))
    assert mean.shape == (3, 1, 1, 3)


def test_recall_constrained_selection_requires_micro_and_instance_macro():
    rows = {
        "whole_domain": [
            {
                "percentile": 50.0,
                "threshold": 1.0,
                "train_hairpin_recall": 0.96,
                "train_vortex_equal_recall": 0.90,
                "candidate_fraction": 0.10,
            },
            {
                "percentile": 40.0,
                "threshold": 0.8,
                "train_hairpin_recall": 0.97,
                "train_vortex_equal_recall": 0.96,
                "candidate_fraction": 0.20,
            },
        ],
        "wall_normal_plane": [
            {
                "percentile": 80.0,
                "threshold": 2.0,
                "train_hairpin_recall": 0.95,
                "train_vortex_equal_recall": 0.95,
                "candidate_fraction": 0.15,
            }
        ],
    }
    selection, annotated = select_recall_constrained_ivd(
        rows,
        minimum_train_hairpin_recall=0.95,
        minimum_train_vortex_equal_recall=0.95,
    )
    assert selection.mode == "wall_normal_plane"
    assert selection.threshold == 2.0
    rejected = next(
        row
        for row in annotated
        if row["mode"] == "whole_domain" and row["percentile"] == 50.0
    )
    assert not rejected["meets_recall_constraint"]


def test_four_vortex_type_rules_keep_neck_in_limb():
    velocity = np.tile(np.asarray([[1.0, 0.0, 0.0]]), (4, 1))
    vorticity = np.asarray(
        [
            [2.0, 0.0, 0.0],  # ordinary streamwise
            [0.0, 2.0, 0.0],  # ordinary spanwise
            [0.1, 2.0, 0.2],  # positive y-dominant hairpin head
            [0.1, 0.2, 2.0],  # vertical hairpin neck -> limb
        ]
    )
    result = build_four_vortex_type_labels(
        velocity,
        vorticity,
        omega_y_prime=np.asarray([0.0, 2.0, 2.0, 0.2]),
        vortex_ids=np.asarray([0, 0, 7, 7]),
    )
    assert result["labels"].tolist() == [
        ORDINARY_STREAMWISE,
        ORDINARY_SPANWISE,
        HAIRPIN_HEAD,
        HAIRPIN_LIMB,
    ]


def test_cluster_mapping_is_frozen_by_permutation():
    truth = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    clusters = np.asarray([2, 2, 0, 0, 3, 3, 1, 1])
    mapping, score = best_cluster_permutation(truth, clusters)
    predicted = apply_cluster_mapping(clusters, mapping)
    metrics = four_class_metrics(truth, predicted)
    assert score == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["adjusted_rand_index"] == 1.0


def test_cluster_permutation_accepts_cluster_absent_from_subset():
    truth = np.asarray([0, 1, 2, 3])
    clusters = np.asarray([0, 0, 1, 2])
    mapping, score = best_cluster_permutation(
        truth, clusters, cluster_ids_universe=(0, 1, 2, 3)
    )
    assert set(mapping) == {0, 1, 2, 3}
    assert 0.0 <= score <= 1.0


def test_buffered_x_split_keeps_complete_ids_and_excludes_crossers():
    vortex_ids = np.zeros((1, 1, 8), dtype=np.int32)
    vortex_ids[0, 0, 1:3] = 1
    vortex_ids[0, 0, 3:5] = 2  # crosses the excluded middle buffer
    vortex_ids[0, 0, 5:7] = 3
    x_centers = (np.arange(8, dtype=np.float64) + 0.5) / 8.0
    splits, excluded = split_vortex_ids_by_buffered_x(
        vortex_ids,
        x_centers,
        x_intervals={"train": (0.10, 0.40), "test": (0.60, 0.90)},
        period_x=1.0,
        domain_min_x=0.0,
    )
    assert splits["train"].tolist() == [1]
    assert splits["test"].tolist() == [3]
    assert excluded.tolist() == [2]
