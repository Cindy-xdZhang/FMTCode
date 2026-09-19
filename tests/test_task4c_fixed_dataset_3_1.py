"""Dense v3 must preserve v2 sampling decisions, geometry and instance folds."""
import json
from pathlib import Path
import unittest
import numpy as np
from FMT_Utils import Task4C_FixedDataset_2_1 as v2
from FMT_Utils import Task4C_FixedDataset_3_1 as v3
from experiments.Task4C_FixedDataset_3_1 import validate_spec


class DenseDatasetRules(unittest.TestCase):
    def test_sparse_and_dense_accept_exactly_the_same_ordered_points(self):
        rng = np.random.default_rng(193)
        for scale in ([1., 1., 1.], [3., .1, 1.]):
            points = rng.random((18000, 3))*scale
            points = np.concatenate([points, points[:200]])
            for radius in (.025, .06, .13):
                old = v2.dart_throw(points, radius, 3500)
                new = v3.dart_throw(points, radius, 3500)
                np.testing.assert_array_equal(old, new)

    def test_sparse_cells_handle_extent_that_exceeds_dense_grid_limit(self):
        points = np.array([[0., 0., 0.], [1000., 1000., 1000.], [.01, .01, .01], [1., 1., 1.]])
        with self.assertRaises(ValueError):
            v2.dart_throw(points, .1, 4)
        np.testing.assert_array_equal(v3.dart_throw(points, .1, 4), [0, 1, 3])

    def test_radius_search_and_all_selected_ids_are_identical(self):
        points = np.random.default_rng(21).random((10000, 3))
        a, r, log = v2.poisson_disk(points, 2000, .06)
        b, s, log2 = v3.poisson_disk(points, 2000, .06)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(r, s)
        self.assertEqual(log, log2)

    def test_exact_distance_boundary_and_duplicates(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [1., 0., 0.], [2., 0., 0.], [.5, 0., 0.]])
        np.testing.assert_array_equal(v3.dart_throw(points, 1., 5), [0, 1, 3])
        np.testing.assert_array_equal(v3.dart_throw(points, 1., 5), v2.dart_throw(points, 1., 5))

    def test_frozen_folds_do_not_move_when_counts_change(self):
        ids = [0, 1, 2]
        low = np.array([[0., 0., 0.], [.5, 0., 0.], [3., 0., 0.]])
        reference = dict(fold_of_instance={'0': 1, '1': 1, '2': 0},
                         groups=[dict(group=[0, 1], fold=1), dict(group=[2], fold=0)])
        mapping, report = v3.fixed_instance_folds(ids, low, low+1, {0: 3000, 1: 1, 2: 20}, 5, 98303, reference)
        self.assertEqual(mapping, {0: 1, 1: 1, 2: 0})
        self.assertEqual(report['fold_samples'], [20, 3001, 0, 0, 0])

    def test_only_density_changes_and_no_training_phase(self):
        spec = json.loads(Path('config/mainExp_Task4C_FixedDataset_3.1.json').read_text())
        validate_spec(spec)
        self.assertEqual(spec['sampling']['pool_per_flow']*2, 40_000_000)
        self.assertEqual(spec['sampling']['samples_per_flow']*2, 8_000_000)
        self.assertIs(v3.adapter().trace_curves, v2.trace_curves)
        self.assertIs(v3.adapter().cut_and_clean, v2.cut_and_clean)
        self.assertIs(v3.adapter().initial_pool, v2.initial_pool)
        self.assertIs(v3.adapter().assign_instances, v2.assign_instances)


if __name__ == '__main__':
    unittest.main()
