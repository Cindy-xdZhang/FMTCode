"""Full-input, training-fit, split-isolation and durable-evaluation checks."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import build_window,write_json,sha256
from FMT_Utils.FlowMapFit_3D import DirectFMTDecoder,normalization,fit_direct,local_predictions,compose_direct
from experiments.Build_Task678_DirectFMTFit_1_1 import compact,jittered_centers,save_cache,DEFAULT_CONFIG
from experiments.Run_Task678_DirectFMTFit_1_1 import run,load_data
from experiments.Audit_Task678_DirectFMTFit_1_1 import audit
from tests.test_task678_flowmap_3d import analytic_field


class DirectFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_161_input_channels_enter_trainable_layer(self):
        model=DirectFMTDecoder(np.zeros(161),np.ones(161),16,1,1)
        tokens=torch.randn(2,6,161);geometry=torch.randn(2,6,4)
        seen=[]
        hook=model.support[0].register_forward_pre_hook(lambda module,args: seen.append(args[0].detach().clone()))
        q=torch.randn(2,3)
        torch.testing.assert_close(model(tokens,geometry,q,torch.zeros(2),torch.zeros(2,2)),q)
        hook.remove()
        torch.testing.assert_close(seen[0][...,:161],tokens)
        torch.testing.assert_close(seen[0][...,161:],geometry)
        self.assertEqual(model.support[0].in_features,165)
        large=DirectFMTDecoder(np.zeros(161),np.ones(161))
        self.assertGreater(sum(p.numel() for p in large.parameters()),50*55427)

    def test_internal_affine_keeps_constant_and_nonconstant_features(self):
        x=np.random.default_rng(17).normal(size=(12,6,161)).astype(np.float32);x[:,:,3]=17.
        mean,std=normalization(x)
        np.testing.assert_allclose((x-mean)/std*std+mean,x,atol=3e-7)
        self.assertEqual(len(mean),161);self.assertEqual(std[3],1.)
        self.assertTrue(np.isfinite(std).all())

    def test_new_material_seeds_and_within_replica_buffer(self):
        lo,hi=np.zeros(3),np.array([5.,4.,2.])
        a,m=jittered_centers(lo,hi,.03,8,128,20)
        b,_=jittered_centers(lo,hi,.03,8,128,21)
        self.assertFalse(np.array_equal(a,b));self.assertGreater(m['minimum_separation_bound'],.3)
        d=np.linalg.norm(a[:,None]-a[None],axis=-1);np.fill_diagonal(d,np.inf)
        self.assertGreater(d.min(),.3)

    def test_feature_construction_cannot_read_hidden_targets(self):
        d,_=build_window(analytic_field(),np.random.default_rng(2).uniform(-1,1,(4,3)),.1,.5)
        c=compact(d)
        changed={k:v.copy() for k,v in d.items()}
        for k in ('target0','target1','target_long'):
            changed[k][:]=12345.
        z=compact(changed)
        for k in ('support0','support1','context'):
            np.testing.assert_array_equal(c[f'fmt__{k}'],z[f'fmt__{k}'])
        self.assertNotIn('support0',c)

    def test_actual_fit_reduces_known_training_error(self):
        rng=np.random.default_rng(9);n=8;q=4;t=9
        token=np.zeros((n,1,161),np.float32);token[:,0,0]=np.linspace(.2,1.,n)
        query=rng.normal(size=(n,q,3)).astype(np.float32)*.1
        tau=np.linspace(0,1,t,dtype=np.float32)
        velocity=np.stack([token[:,0,0],.2*token[:,0,0],-.1*token[:,0,0]],-1)
        target=query[:,:,None]+tau[None,None,:,None]*velocity[:,None,None]
        arrays=dict(tokens=token,geometry=np.zeros((n,1,4),np.float32),query=query,scales=np.zeros((n,2),np.float32),target=target)
        settings=dict(hidden_width=32,support_blocks=1,query_blocks=1,optimizer_steps=600,bundle_batch=8,
                      queries_per_bundle=8,learning_rate=.003,probe_every=300)
        model,info=fit_direct(arrays,settings,19,'cpu')
        measured=np.sqrt(np.mean(np.sum((local_predictions(model,arrays,'cpu')[:,:,1:]-target[:,:,1:])**2,-1)))
        self.assertLess(measured,.01)
        self.assertLess(measured,info['initial_train_probe_nrmse']*.03)
        self.assertEqual(info['optimizer_steps'],600)

    def test_composition_uses_prediction_not_target_endpoint(self):
        class Translation(torch.nn.Module):
            def encode_support(self,tokens,geometry):
                return tokens[:,0]
            def query_context(self,context,query,tau,scales):
                v=torch.zeros_like(query);v[:,0]=context[:,0]
                return query+tau[:,None]*v
        arrays=dict(tokens=np.array([[[.1]],[[.2]]],np.float32),geometry=np.zeros((2,1,4),np.float32),
            query=np.zeros((2,8,3),np.float32),scales=np.zeros((2,2),np.float32),target=np.full((2,8,5,3),999.,np.float32),
            origin=np.array([[0,0,0],[.7,0,0]],np.float32),radius=np.ones(2,np.float32),duration=np.ones(2,np.float32))
        p,info=compose_direct(Translation(),arrays,'cpu')
        np.testing.assert_allclose(p[:,:,-1,0],.3,atol=1e-6)
        self.assertEqual(info['second_query_uses'],'predicted_first_stage_endpoint')


def smoke(output):
    root=Path(output);root.mkdir(parents=True)
    spec=json.loads(Path(DEFAULT_CONFIG).read_text())
    spec.update(output_root=str(root),datasets=['analytic_unsteady'],training_replicates=2)
    for c in spec['conditions'].values():
        c.update(hidden_width=16,support_blocks=1,query_blocks=1,optimizer_steps=3,probe_every=1,bundle_batch=4,queries_per_bundle=2)
        if c['data']=='memorize':
            c['memorize_regions']=4;c['minimum_steps']=2
    config=root/'smoke_config.json';write_json(config,spec);h=sha256(config);spec['_config_sha256']=h
    records=[]
    for ordinal in range(8):
        role='train' if ordinal<4 else ('validation' if ordinal<6 else 'test')
        for rep in range(2 if role=='train' else 1):
            d,_=build_window(analytic_field(ordinal/8),np.random.default_rng(ordinal*10+rep).uniform(-1,1,(8,3)),.1,.5,query_seed=ordinal*10+rep)
            meta=dict(config_sha256=h,ordinal=ordinal,replicate=rep,role=role,base_cache_sha256='analytic',
                      start_index=ordinal*9,end_index=ordinal*9+8)
            records.append(save_cache(root/'cache/analytic_unsteady'/f'window_{ordinal:02d}_rep{rep:02d}.npz',compact(d),meta))
    write_json(root/'build/analytic_unsteady.json',dict(status='PASS',config_sha256=h,records=records))
    write_json(root/'build/analytic_unsteady_base.json',dict(status='PASS',config_sha256=h,records=[r for r in records if r['replicate']==0]))
    for name in spec['conditions']:
        data,source=load_data(spec,'analytic_unsteady','train',spec['conditions'][name])
        assert all(r['ordinal']<4 for r in source)
        expected=4 if name=='memorize16' else (64 if name=='large_expanded' else 32)
        assert len(data['origin0'])==expected
        run(spec,str(config),'analytic_unsteady',name,9100,'cpu')
    audit(spec,str(config))
    assert json.loads((root/'independent_audit.json').read_text())['metric_records']==30
    # The auditor must reject a numerically changed score even with valid files.
    p=root/'runs/analytic_unsteady/small_base/seed9100/Task6_train.json'
    original=p.read_text();changed=json.loads(original);changed['metrics']['position_nrmse']+=.5;write_json(p,changed)
    try:
        audit(spec,str(config))
    except AssertionError:
        pass
    else:
        raise AssertionError('Audit accepted a corrupted metric')
    finally:
        p.write_text(original,encoding='utf-8')
    print('DIRECT FMT SMOKE PASS: 4 conditions, train/validation/test separation, 30 audited results, corruption rejected',flush=True)


if __name__=='__main__':
    import sys
    if len(sys.argv)==3 and sys.argv[1]=='--smoke-output':
        smoke(sys.argv[2])
    else:
        unittest.main()
