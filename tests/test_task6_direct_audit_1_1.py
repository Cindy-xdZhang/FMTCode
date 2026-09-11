import unittest
import numpy as np
from experiments.Audit_Task6_DirectNeural_1_1 import independent_errors, aggregate_errors, inverse_for_audit_only
from FMT_Utils.Task6DirectNeural_3D import direct_inputs
from tests.test_task6_direct_neural_5_1 import examples


class TestIndependentAudit(unittest.TestCase):
    def test_translation_errors_have_known_physical_units_and_pair_distances(self):
        truth = examples(3).astype(float)
        prediction = truth+np.array([1., 2., 3.])
        scores = aggregate_errors(independent_errors(prediction, truth), np.array([True, False, True]))
        self.assertEqual(scores['samples'], 2)
        for name in ('position_rmse_r', 'all_time_rmse_r', 'center_rmse_r', 'neighbor_rmse_r', 'initial_rmse_r', 'endpoint_rmse_r'):
            self.assertAlmostEqual(scores[name], np.sqrt(14.))
        self.assertLess(scores['pair_distance_rmse_r'], 1e-14)
        np.testing.assert_allclose(scores['time_rmse_r'], np.sqrt(14.))

    def test_independent_reconstruction_of_full_signed_coefficients(self):
        geometry = examples(8)
        recovered = inverse_for_audit_only(direct_inputs(geometry, 'signed_fmt16_neural'))
        np.testing.assert_allclose(recovered, geometry, atol=5e-7)


if __name__ == '__main__':
    unittest.main()
