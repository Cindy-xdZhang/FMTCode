import unittest

import numpy as np

from FMT_Utils.Task4A_PerVortexClustering_3D import (
    fmt_base_feature_width,
    map_clusters_by_proxy_permutation,
    map_clusters_by_geometry_score,
    map_clusters_by_hairpin_topology,
    map_clusters_by_vorticity_axis,
    per_vortex_kmeans,
    velocity_curl_from_time_parameterized_cross,
)


class Task4APerVortexClustering3DTests(unittest.TestCase):
    def test_independent_kmeans_uses_both_clusters_per_instance(self):
        features = np.asarray(
            [[0, 0], [0.1, 0], [5, 0], [5.1, 0],
             [10, 0], [10.1, 0], [20, 0], [20.1, 0]],
            dtype=float,
        )
        ids = np.asarray([1] * 4 + [2] * 4)
        clusters, rows = per_vortex_kmeans(
            features,
            ids,
            base_feature_width=1,
            neighbor_weight=1.0,
            n_init=10,
            seed=3,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(clusters[ids == 1]), {0, 1})
        self.assertEqual(set(clusters[ids == 2]), {0, 1})

    def test_proxy_permutation_is_per_instance_and_explicit(self):
        raw = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
        proxy = np.asarray([1, 1, 2, 2, 2, 2, 1, 1])
        ids = np.asarray([1] * 4 + [2] * 4)
        semantic, rows = map_clusters_by_proxy_permutation(raw, proxy, ids)
        np.testing.assert_array_equal(semantic, proxy)
        self.assertEqual(rows[0]["streamwise_raw_cluster"], 0)
        self.assertEqual(rows[1]["streamwise_raw_cluster"], 1)

    def test_vorticity_axis_mapping_names_each_instance(self):
        raw = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
        ids = np.asarray([1] * 4 + [2] * 4)
        vorticity = np.asarray(
            [[3, 1, 0], [3, 1, 0], [1, 3, 0], [1, 3, 0],
             [1, 3, 0], [1, 3, 0], [3, 1, 0], [3, 1, 0]],
            dtype=float,
        )
        semantic, rows = map_clusters_by_vorticity_axis(raw, ids, vorticity)
        np.testing.assert_array_equal(semantic, [1, 1, 2, 2, 2, 2, 1, 1])
        self.assertEqual(rows[0]["streamwise_raw_cluster"], 0)
        self.assertEqual(rows[1]["streamwise_raw_cluster"], 1)

    def test_geometry_score_mapping_is_proxy_independent(self):
        raw = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
        ids = np.asarray([1] * 4 + [2] * 4)
        score = np.asarray([0.9, 0.8, 0.1, 0.2, 0.1, 0.2, 0.8, 0.9])
        semantic, rows = map_clusters_by_geometry_score(raw, ids, score)
        np.testing.assert_array_equal(semantic, [1, 1, 2, 2, 2, 2, 1, 1])
        self.assertEqual(rows[0]["streamwise_raw_cluster"], 0)
        self.assertEqual(rows[1]["streamwise_raw_cluster"], 1)

    def test_fmt_base_feature_width(self):
        self.assertEqual(fmt_base_feature_width(6, "gram", True), 23)
        self.assertEqual(fmt_base_feature_width(6, "magnitude", False), 6)

    def test_topology_mapping_selects_two_legs_cluster(self):
        indices = np.asarray(
            [
                [0, 0, 0], [0, 0, 1],
                [4, 0, 0], [4, 0, 1],
                [1, 0, 2], [2, 0, 2], [3, 0, 2],
            ],
            dtype=np.int32,
        )
        raw = np.asarray([0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
        ids = np.ones(len(raw), dtype=np.int32)
        semantic, rows = map_clusters_by_hairpin_topology(
            raw,
            ids,
            indices,
            dominant_min_voxels=2,
            dominant_min_fraction=0.0,
        )
        np.testing.assert_array_equal(semantic, [1, 1, 1, 1, 2, 2, 2])
        self.assertEqual(rows[0]["streamwise_raw_cluster"], 0)

    def test_cross_geometry_reconstructs_linear_field_curl(self):
        offset = 0.1
        dt = 0.01
        initial = np.asarray(
            [
                [0, 0, 0], [-offset, 0, 0], [offset, 0, 0],
                [0, -offset, 0], [0, offset, 0],
                [0, 0, -offset], [0, 0, offset],
            ],
            dtype=float,
        )

        def velocity(points):
            # v=(1-y, x, 0) has curl=(0,0,2).
            return np.column_stack((1.0 - points[:, 1], points[:, 0], np.zeros(len(points))))

        line_velocity = velocity(initial)
        primitive = np.stack(
            (initial - dt * line_velocity, initial, initial + dt * line_velocity),
            axis=1,
        )[None]
        reconstructed = velocity_curl_from_time_parameterized_cross(
            primitive, parameter_step=dt, cross_offset=offset
        )
        np.testing.assert_allclose(reconstructed["velocity_xyz"][0], [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(reconstructed["curl_xyz"][0], [0, 0, 2], atol=1e-6)
        self.assertAlmostEqual(float(reconstructed["angle_degrees"][0]), 90.0)


if __name__ == "__main__":
    unittest.main()
