import unittest
from unittest.mock import patch
import numpy as np

from FMT_Utils import Task4C_V2NewLabel_1_1 as data


class CurveLabelsTest(unittest.TestCase):
    def test_strict_majority_and_instance_zero(self):
        owners = np.full((4, 32), -1, np.int64)
        owners[1, :16] = 0
        owners[2, :17] = 0
        owners[3, :9] = 0
        owners[3, 9:17] = 7
        labels, hits = data.majority_labels(owners)
        np.testing.assert_array_equal(labels, [0, 0, 1, 1])
        np.testing.assert_array_equal(hits, [0, 16, 17, 17])

    def test_shortest_only_with_pointwise_gt_query(self):
        curves = np.full((3, 3, 32, 3), -1, np.float32)
        curves[0, 0, :17, 0] = 1
        curves[1, 0, :16, 0] = 1
        curves[1, 1:] = 1
        curves[2, 0, :, 0] = 1
        def exact_query(gt, points, locator):
            return np.where(points[:, 0] > 0, 0, -1), locator
        with patch.object(data, 'sample_gt', side_effect=exact_query) as query:
            result = data.shortest_curve_labels(None, curves, batch=2)
        self.assertEqual(query.call_count, 2)
        np.testing.assert_array_equal(result['label'], [1, 0, 1])
        np.testing.assert_array_equal(result['short_curve_gt_count'], [17, 16, 32])
        np.testing.assert_array_equal(result['short_curve_gt_fraction'], [17/32, .5, 1.])

    def test_requires_saved_geometry_and_32_points(self):
        with self.assertRaises(ValueError): data.majority_labels(np.zeros((1, 31), np.int64))
        with self.assertRaises(ValueError): data.majority_labels(np.zeros((1, 32), np.float64))
        with self.assertRaises(ValueError): data.shortest_curve_labels(None, np.zeros((1, 3, 32, 3), np.float64))

    def test_actual_vtk_cells_and_nonzero_outside_coordinates(self):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
        grid = vtk.vtkImageData()
        grid.SetOrigin(10, 20, 30); grid.SetSpacing(1, 1, 1); grid.SetDimensions(2, 2, 2)
        ids = numpy_to_vtk(np.array([0], np.int64)); ids.SetName('VortexIds')
        grid.GetCellData().AddArray(ids)
        curves = np.tile(np.array([15, 25, 35], np.float32), (3, 3, 32, 1))
        for row, hits in enumerate((15, 16, 17)):
            curves[row, 0, :hits] = [10.5, 20.5, 30.5]
        result = data.shortest_curve_labels(grid, curves, batch=1)
        np.testing.assert_array_equal(result['label'], [0, 0, 1])
        np.testing.assert_array_equal(result['short_curve_gt_count'], [15, 16, 17])

    def test_balance_correction_targets_one_positive_two_negative(self):
        p, survival = data.balanced_fraction([600, 300], [480, 285])
        self.assertAlmostEqual(p*survival[1]/((1-p)*survival[0]), .5)
        self.assertLess(p, 1/3)

    def test_frozen_v2_parameters_are_checked(self):
        import json
        from pathlib import Path
        from experiments.Task4C_V2NewLabel_1_1 import validate_spec, CONFIG
        spec = json.loads(Path(CONFIG).read_text())
        validate_spec(spec)
        spec['sampling']['samples_per_flow'] += 1
        with self.assertRaises(AssertionError): validate_spec(spec)


if __name__ == '__main__':
    unittest.main()
