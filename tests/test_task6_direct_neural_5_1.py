import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import cross_offsets
from FMT_Utils.Task6DirectNeural_3D import (
    DirectGeometryVAE, audit_direct_model, direct_inputs, fit_statistics,
    predict_direct, train_direct,
)
from FMT_Utils.Task6PCABaseline_3D import GeometryPCABaseline
from FMT_Utils.Task6Recovery_3D import signed_fmt


def examples(n, seed=51):
    rng = np.random.default_rng(seed)
    x = np.broadcast_to(cross_offsets()[None, :, None], (n, 7, 32, 3)).copy()
    t = np.linspace(0, 1, 32)[None, None, :, None]
    x += rng.normal(size=(n, 1, 1, 3)) * t
    x += rng.normal(size=(n, 1, 1, 3)) * t * t * .2
    return x.astype(np.float32)


class TestDirectNeural(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_forward_features_match_frozen_coefficients_and_raw_constants(self):
        y = examples(10)
        np.testing.assert_array_equal(direct_inputs(y, 'signed_fmt16_neural'), signed_fmt(y, 16).astype(np.float32))
        np.testing.assert_array_equal(direct_inputs(y, 'raw_neural'), y[:, :, 1:].reshape(10, 651))

    def test_equal_parameter_initialization_and_no_hidden_fixed_matrix(self):
        y = examples(32)
        models = []
        for arm in ('raw_neural', 'signed_fmt16_neural'):
            x = direct_inputs(y, arm)
            torch.manual_seed(77)
            models.append(DirectGeometryVAE(fit_statistics(x, y), 12, 32, 1, .1))
        for a, b in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(audit_direct_model(models[0])['trainable_parameters'],
                         audit_direct_model(models[1])['trainable_parameters'])
        for model in models:
            self.assertEqual(set(dict(model.named_buffers())),
                             {'input_mean', 'input_std', 'output_mean', 'output_scale', 'initial_seeds'})

    def test_statistics_are_train_only_and_do_not_modify_arrays(self):
        y = examples(20)
        original = y.copy()
        x = direct_inputs(y[:12], 'signed_fmt16_neural')
        stats = fit_statistics(x, y[:12])
        np.testing.assert_array_equal(y, original)
        np.testing.assert_allclose(stats['output_mean'], y[:12, :, 1:].mean(0, dtype=np.float64).reshape(651))
        y[12:, :, 1:] += 1000
        again = fit_statistics(x, y[:12])
        for key in stats:
            np.testing.assert_array_equal(stats[key], again[key])

    def test_zero_parameters_remove_every_input_dependent_reconstruction(self):
        y = examples(20)
        x = direct_inputs(y, 'signed_fmt16_neural')
        model = DirectGeometryVAE(fit_statistics(x, y), 12, 32, 1, 0)
        for parameter in model.parameters():
            parameter.data.zero_()
        prediction = predict_direct(model, x, torch.device('cpu'))
        np.testing.assert_allclose(prediction[:, :, 1:],
            np.broadcast_to(y[:, :, 1:].mean(0), (len(y), 7, 31, 3)), atol=2e-7)
        np.testing.assert_allclose(prediction, np.broadcast_to(prediction[:1], prediction.shape), atol=0)

    def test_learning_uses_latent_without_inverse_or_pca_calls(self):
        y = examples(32)
        cfg = dict(width=64, blocks=1, latent_dim=12, batch_size=32, updates=450,
                   probe_every=150, beta=1e-5, gradient_clip=1.)
        candidate = dict(id='test', learning_rate=.003, dropout=0., weight_decay=0.)
        with tempfile.TemporaryDirectory() as folder:
            with patch('numpy.fft.irfft', side_effect=AssertionError('Inverse forbidden')), \
                 patch('numpy.linalg.eigh', side_effect=AssertionError('PCA forbidden')), \
                 patch('numpy.linalg.svd', side_effect=AssertionError('PCA forbidden')):
                model, result = train_direct(y, y, 'signed_fmt16_neural', candidate, cfg, 91,
                    torch.device('cpu'), Path(folder) / 'fit')
            self.assertLess(result['train_rmse_r'], .1 * result['curve'][0]['train_rmse_r'])
            self.assertGreater(result['train_zero_latent_rmse_r'], 5 * result['train_rmse_r'])
            x = torch.tensor(direct_inputs(y, 'signed_fmt16_neural'))
            model.eval()
            prediction, mean, _ = model(x, sample=False)
            torch.testing.assert_close(prediction, model.decode(mean), rtol=0, atol=0)
            self.assertEqual(result['statistics_train_samples'], 32)
            self.assertGreater(result['selected_step'], 0)
            self.assertFalse(list(Path(folder).rglob('*.pt')))

    def test_independent_pca_compresses_without_modifying_geometry(self):
        y = examples(40)
        original = y.copy()
        model = GeometryPCABaseline(y, 8)
        query = examples(12, seed=78)
        self.assertEqual(model.encode(query).shape, (12, 8))
        np.testing.assert_allclose(model.predict(query), query, atol=2e-6)
        np.testing.assert_array_equal(y, original)


if __name__ == '__main__':
    unittest.main()
