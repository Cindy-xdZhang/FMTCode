"""Frozen FMT identity, routing, equal-capacity controls and actual learning."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from FMT_Utils.PrimitiveVAE_3D import fmt_tokens
from FMT_Utils.Task6FMTGeometryMoE_3D import FMTGeometryMoE,original_fmt,train_moe,fit_statistics,cached,audit_model
from tests.test_task6_direct_neural_5_1 import examples


def candidate(family='raw_mlp',gate='scalar'):
    return dict(id='fixture_'+family,geometry_branch=family,gate=gate,dropout=.02,weight_decay=0.,
        learning_rate=.003,scheduler=dict(name='cosine_legacy'),auxiliary_weight=.05,
        architecture=dict(latent_dim=12,expert_width=32,expert_blocks=2,decoder_width=48,decoder_blocks=2,
                          temporal_width=16,temporal_heads=2,temporal_layers=2,gate_width=16))


class TestMoE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_fmt_exactly_161_and_batch_independent(self):
        y=examples(11,1)
        x=torch.tensor(y)
        out=original_fmt(x)
        self.assertEqual(out.shape,(11,161))
        np.testing.assert_array_equal(out.numpy(),fmt_tokens(y))
        torch.testing.assert_close(out,torch.cat([original_fmt(x[:4]),original_fmt(x[4:])]),rtol=0,atol=0)

    def test_capacity_control_and_shared_initialization(self):
        y=examples(6,2)
        for family in ('raw_mlp','pointnn'):
            models=[]
            for arm in ('fmt_moe','geometry_moe','geometry_only'):
                torch.manual_seed(17)
                models.append(FMTGeometryMoE(fit_statistics(y),candidate(family),arm))
            self.assertEqual(audit_model(models[0])['trainable_parameters'],audit_model(models[1])['trainable_parameters'])
            for key,value in models[2].state_dict().items():
                torch.testing.assert_close(value,models[0].state_dict()[key],rtol=0,atol=0)
            a,b=models[0].features(torch.tensor(y))
            torch.testing.assert_close(a[:,:161],original_fmt(torch.tensor(y)),rtol=0,atol=0)
            self.assertTrue((a[:,161:]==0).all())
            self.assertEqual(sum(p.numel() for p in models[0].frontend.parameters()),0)

    def test_routing_endpoints_and_conditioning_use_training_only(self):
        y=examples(8,3)
        for gate in ('scalar','channel'):
            m=FMTGeometryMoE(fit_statistics(y),candidate(gate=gate),'fmt_moe')
            features=cached(m,y,torch.device('cpu'))
            m.condition(features)
            mean=m.a_mean.clone()
            _=m.features(torch.tensor(y)*100)
            torch.testing.assert_close(mean,m.a_mean)
            m.eval()
            z,aux=m.encode_features(features)
            torch.testing.assert_close(aux['weight_a'],torch.full_like(aux['weight_a'],.1))
            torch.testing.assert_close(z,.1*aux['a']+.9*aux['b'])
            only_b,_=m.encode_features(features,'geometry')
            only_a,_=m.encode_features(features,'expert_a')
            torch.testing.assert_close(only_b,aux['b'])
            torch.testing.assert_close(only_a,aux['a'])
            prediction=m.decode(z)
            self.assertEqual(prediction.shape,(8,7,32,3))
            torch.testing.assert_close(prediction[:,:,0],torch.tensor(y[:,:,0]),rtol=0,atol=0)

    def test_both_geometry_families_fit_and_keep_latent_relevant(self):
        y=examples(24,4)
        training=dict(updates=350,batch_size=8,warmup_updates=20,probe_every=175,gradient_clip=1.)
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            for family in ('raw_mlp','pointnn'):
                model,fit=train_moe(y,y,'fmt_moe',candidate(family),training,27,torch.device('cpu'),Path(temp)/family)
                self.assertLess(fit['train_rmse_r'],fit['curve'][0]['train_rmse_r']*.2)
                self.assertGreater(fit['train_zero_latent_rmse_r'],fit['train_rmse_r']*2)
                self.assertTrue(np.isfinite(list(fit['validation_branch_diagnostics'].values())).all())
                self.assertEqual(fit['train_examples_exposed'],350*8)
                # All trainable modules receive gradients after the zero-output initialization.
                for name,param in model.named_parameters():
                    self.assertIsNotNone(param.grad,name)
                self.assertEqual(fit['structure']['decoder_input'],'mixed latent only')
            self.assertFalse(list(Path(temp).rglob('*.pt')))


if __name__=='__main__':
    unittest.main()
