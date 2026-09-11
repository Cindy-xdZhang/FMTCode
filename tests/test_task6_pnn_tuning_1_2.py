"""Tuning contracts: fixed front end, train-only conditioning and validation gates."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from FMT_Utils.Task6PNNTrans_3D import PNNTransAutoencoder, fit_statistics
from FMT_Utils.Task6PNNTuning_3D import (TunedPNNAutoencoder,condition_features,
    mixed_geometry,audit_tuned,train_tuned)
from experiments.Task6_PNNTuning_1_2 import promotion_ranking,selection_result
from tests.test_task6_direct_neural_5_1 import examples


def candidate():
    return dict(id='fixture',architecture=dict(width=16,heads=2,layers=2,latent_dim=8,
        decoder_width=32,decoder_blocks=2,frequencies=4,alpha=1000.,beta=1.),
        learning_rate=.003,dropout=0.,weight_decay=0.,input_mlp=True,
        feature_conditioning='floored_standardization',input_dropout=0.,feature_noise=0.,geometry_mix_fraction=.5)


class TestTuning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frozen_frontend_conditioning_and_eval_noise(self):
        y=examples(8,1)
        c=candidate()
        c.update(dropout=.4,input_dropout=.2,feature_noise=.1)
        model=TunedPNNAutoencoder(fit_statistics(y),'pnn_trans',c)
        old=PNNTransAutoencoder(fit_statistics(y),**c['architecture'])
        features=model.snapshot_features(torch.tensor(y))
        torch.testing.assert_close(features,old.snapshot_features(torch.tensor(y)),rtol=0,atol=0)
        mean,scale=condition_features(features,c['feature_conditioning'])
        model.feature_mean.copy_(mean)
        model.feature_scale.copy_(scale)
        self.assertGreaterEqual(float(scale.min()),.001)
        audit_tuned(model)
        # Fitting the conditioner does not need validation or test inputs.
        snapshot=mean.clone()
        _=model.snapshot_features(torch.tensor(y)*100)
        torch.testing.assert_close(model.feature_mean,snapshot)
        model.eval()
        a=model.encode(torch.tensor(y))
        torch.testing.assert_close(a,model.encode(torch.tensor(y)),rtol=0,atol=0)
        model.train()
        self.assertFalse(torch.allclose(model.encode(torch.tensor(y)),model.encode(torch.tensor(y))))

    def test_geometry_mixing_preserves_seeds_and_reencodes_nonlinearity(self):
        torch.manual_seed(7)
        y=torch.tensor(examples(12,2))
        original=y.clone()
        mixed,count=mixed_geometry(y,.5)
        self.assertEqual(count,6)
        torch.testing.assert_close(y,original)
        torch.testing.assert_close(mixed[:,:,0],y[:,:,0])
        torch.testing.assert_close(mixed[6:],y[6:])
        self.assertGreater(float((mixed-y).abs().max()),0.)
        self.assertTrue(torch.isfinite(mixed).all())

    def test_promotion_and_gate_require_both_controls(self):
        candidates=[dict(id='a'),dict(id='b')]
        baseline=[dict(dataset=d,validation_rmse_r=1.) for d in ('x','y')]
        spec=dict(candidates=candidates,screen_datasets=['x','y'],promote_count=1,datasets=['x','y'],
                  validation_gate_rmse_r=3.,selection='fixture')
        screen=[dict(dataset=d,candidate=c,validation_rmse_r=v) for d in spec['datasets']
                for c,v in [('a',.5),('b',.8)]]
        scores,selected=promotion_ranking(screen,baseline,spec)
        self.assertEqual(selected[0]['id'],'a')
        rows=[dict(dataset=d,candidate='a',arm=arm,validation_rmse_r=v)
              for d in spec['datasets'] for arm,v in [('pnn_trans',.5),('raw_trans',.4)]]
        self.assertFalse(selection_result(rows,baseline,selected,spec)['gate_passed'])
        for r in rows:
            if r['arm']=='raw_trans':
                r['validation_rmse_r']=.7
        self.assertTrue(selection_result(rows,baseline,selected,spec)['gate_passed'])
        rows[0]['validation_rmse_r']=3.1
        self.assertFalse(selection_result(rows,baseline,selected,spec)['gate_passed'])

    def test_actual_regularized_training_keeps_latent_relevant(self):
        y=examples(16,3)
        c=candidate()
        c.update(dropout=.03,input_dropout=.01,feature_noise=.001)
        training=dict(updates=450,batch_size=16,warmup_updates=20,probe_every=225,gradient_clip=1.)
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            for arm in ('pnn_trans','raw_trans'):
                model,fit=train_tuned(y,y,arm,c,training,17,torch.device('cpu'),Path(temp)/arm)
                self.assertLess(fit['train_rmse_r'],.25*fit['curve'][0]['train_rmse_r'])
                self.assertGreater(fit['train_zero_latent_rmse_r'],fit['train_rmse_r']*2)
                self.assertEqual(fit['synthetic_geometries_exposed'],450*8)
                self.assertEqual(fit['train_examples_exposed'],450*16)
            self.assertFalse(list(Path(temp).rglob('*.pt')))


if __name__=='__main__':
    unittest.main()
