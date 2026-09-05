import unittest

import numpy as np

from FMT_Utils.Task4A_StreamlineClustering_3D import (
    ChannelVelocityField3D,
    STREAMWISE_CLASS,
    SPANWISE_CLASS,
    fill_invalid_predictions_within_vortex,
    integrate_bidirectional_cross_primitives,
    stratified_fit_indices,
    summarize_topology,
    topology_proxy_rows,
)


class Task4AStreamlineClustering3DTests(unittest.TestCase):
    def test_bidirectional_constant_field_crosses_periodic_x(self):
        zs = np.linspace(0.0, 1.0, 3)
        ys = np.linspace(0.0, 2.0, 3)
        xs = np.linspace(0.0, 2.0, 3)
        velocity = np.zeros((3, 3, 3, 3), dtype=np.float32)
        velocity[..., 0] = 2.0
        field = ChannelVelocityField3D(
            axes_zyx=(zs, ys, xs),
            velocity_zyx3=velocity,
            wall_bounds_z=(0.0, 1.0),
            periods_xy=(2.0, 2.0),
        )
        primitives, valid = integrate_bidirectional_cross_primitives(
            field,
            np.asarray([[1.95, 1.0, 0.5]]),
            spatial_step=0.1,
            steps_per_direction=2,
            offset=0.01,
            chunk_size=1,
        )
        self.assertEqual(primitives.shape, (1, 7, 5, 3))
        self.assertTrue(valid[0])
        np.testing.assert_allclose(
            primitives[0, 0, :, 0], [1.75, 1.85, 1.95, 2.05, 2.15], atol=1e-6
        )

    def test_time_parameterized_integration_retains_speed(self):
        zs = np.linspace(0.0, 1.0, 3)
        ys = np.linspace(0.0, 2.0, 3)
        xs = np.linspace(0.0, 2.0, 3)
        velocity = np.zeros((3, 3, 3, 3), dtype=np.float32)
        velocity[..., 0] = 2.0
        field = ChannelVelocityField3D(
            axes_zyx=(zs, ys, xs),
            velocity_zyx3=velocity,
            wall_bounds_z=(0.0, 1.0),
            periods_xy=(2.0, 2.0),
        )
        primitives, valid = integrate_bidirectional_cross_primitives(
            field,
            np.asarray([[1.95, 0.5, 0.5]]),
            spatial_step=0.1,
            steps_per_direction=1,
            offset=0.05,
            minimum_speed=1e-9,
            unit_velocity=False,
        )
        self.assertTrue(valid.all())
        np.testing.assert_allclose(
            primitives[0, 0, :, 0], [1.75, 1.95, 2.15], atol=1e-6
        )

    def test_stratified_fit_indices_caps_each_vortex(self):
        ids = np.asarray([1] * 9 + [2] * 3 + [3] * 7)
        chosen = stratified_fit_indices(ids, max_per_vortex=4, seed=13)
        counts = {value: int(np.count_nonzero(ids[chosen] == value)) for value in (1, 2, 3)}
        self.assertEqual(counts, {1: 4, 2: 3, 3: 4})

    def test_invalid_predictions_are_filled_only_within_instance(self):
        prediction = np.asarray([STREAMWISE_CLASS, 0, SPANWISE_CLASS, 0], dtype=np.int8)
        valid = np.asarray([True, False, True, False])
        ids = np.asarray([1, 1, 2, 2])
        indices = np.asarray([[0, 0, 0], [1, 0, 0], [10, 0, 0], [11, 0, 0]])
        filled, metadata = fill_invalid_predictions_within_vortex(
            prediction, valid, ids, indices
        )
        np.testing.assert_array_equal(
            filled, [STREAMWISE_CLASS, STREAMWISE_CLASS, SPANWISE_CLASS, SPANWISE_CLASS]
        )
        self.assertEqual(metadata["nearest_filled_voxel_count"], 2)
        self.assertEqual(metadata["vortex_ids_without_valid_primitive"], [])

    def test_topology_proxy_recognizes_exact_two_plus_one(self):
        instances = np.ones((1, 3, 5), dtype=np.int32)
        classes = np.full(instances.shape, STREAMWISE_CLASS, dtype=np.int8)
        classes[:, :, 2] = SPANWISE_CLASS
        rows = topology_proxy_rows(
            instances,
            classes,
            dominant_min_voxels=3,
            dominant_min_fraction=0.05,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["streamwise_dominant_component_count"], 2)
        self.assertEqual(rows[0]["spanwise_dominant_component_count"], 1)
        self.assertTrue(rows[0]["hard_2plus1_success"])
        self.assertAlmostEqual(rows[0]["soft_topology_score"], 1.0)
        summary = summarize_topology(rows)
        self.assertAlmostEqual(summary["hard_2plus1_success_rate_all"], 1.0)


if __name__ == "__main__":
    unittest.main()
