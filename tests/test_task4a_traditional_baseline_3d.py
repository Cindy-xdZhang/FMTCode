import unittest

import numpy as np

from FMT_Utils.Task4A_TraditionalBaseline_3D import (
    paired_topology_comparison,
    proxy_groundtruth_agreement,
    velocity_curl_orientation_classes,
)


class Task4ATraditionalBaseline3DTests(unittest.TestCase):
    def test_parallel_perpendicular_and_axial_sign(self):
        root_half = np.sqrt(0.5)
        velocity = np.asarray([[1, 0, 0]] * 5, dtype=float)
        curl = np.asarray(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [root_half, root_half, 0],
                [0.5, np.sqrt(0.75), 0],
            ],
            dtype=float,
        )
        classes, angles, valid = velocity_curl_orientation_classes(velocity, curl)
        np.testing.assert_array_equal(classes, [1, 1, 2, 1, 2])
        np.testing.assert_allclose(angles, [0, 0, 90, 45, 60], atol=1e-5)
        self.assertTrue(valid.all())

    def test_zero_vector_is_unassigned(self):
        classes, angles, valid = velocity_curl_orientation_classes(
            np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float),
            np.asarray([[1, 0, 0], [0, 0, 0]], dtype=float),
        )
        np.testing.assert_array_equal(classes, [0, 0])
        self.assertTrue(np.isnan(angles).all())
        self.assertFalse(valid.any())

    def test_paired_summary_preserves_instance_identity(self):
        fmt = [
            {"vortex_id": 1, "voxel_count": 10, "input_single_component": True,
             "streamwise_dominant_component_count": 1,
             "spanwise_dominant_component_count": 1,
             "hard_2plus1_success": False, "soft_topology_score": 0.2},
            {"vortex_id": 2, "voxel_count": 12, "input_single_component": True,
             "streamwise_dominant_component_count": 2,
             "spanwise_dominant_component_count": 1,
             "hard_2plus1_success": True, "soft_topology_score": 1.0},
        ]
        baseline = [
            {**fmt[0], "streamwise_dominant_component_count": 2,
             "hard_2plus1_success": True, "soft_topology_score": 1.0},
            {**fmt[1], "soft_topology_score": 0.8},
        ]
        rows, summary = paired_topology_comparison(fmt, baseline)
        self.assertEqual([row["vortex_id"] for row in rows], [1, 2])
        self.assertEqual(summary["baseline_soft_score_better_count"], 1)
        self.assertEqual(summary["baseline_soft_score_worse_count"], 1)
        self.assertEqual(summary["baseline_hard_success_count"], 2)

    def test_proxy_metrics_equal_weight_vortex_ids(self):
        proxy = np.asarray([1, 2, 1, 1, 1, 1], dtype=np.int8)
        fmt = np.asarray([1, 2, 2, 2, 2, 2], dtype=np.int8)
        vortex_ids = np.asarray([1, 1, 2, 2, 2, 2], dtype=np.int32)
        angles = np.asarray([0, 90, 0, 0, 0, 0], dtype=float)
        rows, summary = proxy_groundtruth_agreement(
            fmt, proxy, vortex_ids, angles
        )
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["macro_f1"], 1.0)
        self.assertAlmostEqual(rows[1]["macro_f1"], 0.0)
        self.assertAlmostEqual(
            summary["primary_vortex_equal_weighted_macro_f1"]["mean"], 0.5
        )
        self.assertAlmostEqual(
            summary["vortex_equal_weighted_hard_label_mse"]["mean"], 0.5
        )
        self.assertNotAlmostEqual(
            summary["voxel_weighted_descriptive_metrics"]["hard_label_mse"], 0.5
        )

    def test_proxy_metric_empty_class_agreement_is_one(self):
        rows, summary = proxy_groundtruth_agreement(
            np.asarray([1, 1], dtype=np.int8),
            np.asarray([1, 1], dtype=np.int8),
            np.asarray([7, 7], dtype=np.int32),
            np.asarray([0, 10], dtype=float),
        )
        self.assertAlmostEqual(rows[0]["f1_spanwise"], 1.0)
        self.assertAlmostEqual(rows[0]["macro_f1"], 1.0)
        self.assertAlmostEqual(
            summary[
                "secondary_vortex_equal_weighted_angle_confidence_weighted_mse"
            ]["mean"],
            0.0,
        )

    def test_angle_confidence_downweights_threshold_boundary(self):
        rows, _ = proxy_groundtruth_agreement(
            np.asarray([2, 1], dtype=np.int8),
            np.asarray([1, 2], dtype=np.int8),
            np.asarray([1, 1], dtype=np.int32),
            np.asarray([45, 90], dtype=float),
        )
        self.assertAlmostEqual(rows[0]["hard_label_mse"], 1.0)
        self.assertAlmostEqual(rows[0]["angle_confidence_weighted_mse"], 1.0)


if __name__ == "__main__":
    unittest.main()
