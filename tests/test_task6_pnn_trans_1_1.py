"""Numerical and training contracts for the Point-NN temporal autoencoder."""
import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from scipy.special import erf

from FMT_Utils.Task6PNNTrans_3D import (SnapshotPointNN, PNNTransAutoencoder,
    audit_model, fit_statistics, train_model)
from tests.test_task6_direct_neural_5_1 import examples


class TestPNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_independent_numpy_operator_reference(self):
        q = np.random.default_rng(22).normal(size=(2, 3, 7, 3))
        def pe(x, m):
            frequency = 1/1000.**(np.arange(m)/m)
            phase = x[..., None]*frequency
            return np.stack((np.sin(phase), np.cos(phase)), axis=-1).reshape(*x.shape[:-1], 6*m)
        f = pe(q, 8)
        dp = q[..., None, :, :]-q[..., :, None, :]
        df = f[..., None, :, :]-f[..., :, None, :]
        e = pe(dp/(dp.std(axis=(-3,-2,-1), ddof=1, keepdims=True)+1e-5), 16)
        h = np.concatenate((df/(df.std(axis=(-3,-2,-1), ddof=1, keepdims=True)+1e-5),
                            np.broadcast_to(f[..., :, None, :], df.shape)), axis=-1)
        w = (h+e)*e
        pooled = (w.max(-2)+w.mean(-2))/math.sqrt(1+1e-5)
        pooled = pooled*.5*(1+erf(pooled/math.sqrt(2)))
        expected = np.concatenate((f, pooled), axis=-1).reshape(2,3,1008)
        module = SnapshotPointNN(np.zeros(3), 1.).double()
        got = module(torch.tensor(q)).numpy()
        np.testing.assert_allclose(got, expected, rtol=3e-6, atol=3e-6)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 0)

    def test_no_batch_or_temporal_dependency_and_identity_preserved(self):
        y = torch.tensor(examples(3, 19)).transpose(1, 2)
        module = SnapshotPointNN(np.zeros(3), 2.)
        output = module(y)
        torch.testing.assert_close(output[0], module(y[:1])[0], rtol=0, atol=0)
        reverse = torch.arange(31,-1,-1)
        torch.testing.assert_close(module(y[:, reverse]), output[:, reverse])
        perm = torch.tensor([0,2,1,3,4,5,6])
        permuted = module(y[:,:,perm]).reshape(3,32,7,144)
        torch.testing.assert_close(permuted, output.reshape(3,32,7,144)[:,:,perm], rtol=2e-5, atol=2e-5)
        self.assertGreater(float((permuted.flatten(-2)-output).abs().max()), .01)
        degenerate = torch.zeros(2,4,7,3)
        self.assertTrue(torch.isfinite(module(degenerate)).all())

    def test_latent_only_decode_and_time_attention_gradients(self):
        y = examples(8, 1)
        torch.manual_seed(3)
        model = PNNTransAutoencoder(fit_statistics(y), width=16, heads=2, layers=2,
                                   decoder_width=24, decoder_blocks=1, latent_dim=8, dropout=0.)
        audit = audit_model(model)
        self.assertFalse(audit['analytic_inverse'])
        self.assertEqual(audit['latent_dimension'], 8)
        # After the output layer learns, gradients must reach both temporal attention and input projection.
        torch.nn.init.normal_(model.decoder[-1].weight, std=.01)
        z = model.encode(torch.tensor(y))
        self.assertEqual(tuple(z.shape), (8,8))
        prediction = model.decode(z)
        prediction[:, :, 1:].square().mean().backward()
        self.assertGreater(float(model.temporal_layers[0].self_attn.in_proj_weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.input_projection.weight.grad.abs().sum()), 0.)
        self.assertFalse(torch.allclose(model.encode(torch.tensor(y)[:,:,torch.arange(31,-1,-1)]), z))
        model.eval()
        torch.testing.assert_close(model(torch.tensor(y)), model.decode(model.encode(torch.tensor(y))))

    def test_actual_training_no_checkpoint(self):
        y = examples(12, 7)
        training = dict(architecture=dict(width=16, heads=2, layers=1, latent_dim=8,
            decoder_width=32, decoder_blocks=1, frequencies=4), updates=350,
            batch_size=12, warmup_updates=10, probe_every=175, gradient_clip=1.)
        candidate = dict(id='unit_fixture', dropout=0., learning_rate=.003, weight_decay=0.)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            for arm in ('pnn_trans','raw_trans'):
                _, fit = train_model(y,y,arm,candidate,training,91,torch.device('cpu'),Path(temp)/arm)
                self.assertLess(fit['train_rmse_r'], .15*fit['curve'][0]['train_rmse_r'])
                self.assertGreater(fit['train_zero_latent_rmse_r'], 2*fit['train_rmse_r'])
            self.assertFalse(list(Path(temp).rglob('*.pt')))


if __name__ == '__main__':
    unittest.main()
