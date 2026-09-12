"""Regression checks for loss scaling, nested data growth and final evaluation."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import netCDF4 as nc
import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256, write_json
from FMT_Utils.Task36MultiGate_3D import VARIANTS
from FMT_Utils.Task36Balanced_3D import train_balanced, task_metric_score, build_balanced_model
from FMT_Utils.Task6FMTGeometryMoE_3D import fit_statistics, cached
from FMT_Utils.Task36Labels_3D import load_labels
from experiments import Task36_MultiGate_1_1 as previous
from experiments import Task36_BalancedScale_1_2 as run
from tests.test_task6_direct_neural_5_1 import examples
from tests.test_task36_multigate_3d import joint_candidate


class TestBalancedTask36(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_large_reconstruction_error_cannot_hide_behind_train_variance(self):
        # Recorded 1.1 failure: 6.66 r was reduced to a 0.0125 selection term.
        self.assertGreater(task_metric_score(.9, 6.66), task_metric_score(.4, .93))
        self.assertGreater(task_metric_score(.99, 3.1), task_metric_score(.01, .99))
        self.assertAlmostEqual(task_metric_score(.75, None), .25)
        self.assertAlmostEqual(task_metric_score(None, .25), .25)

    def test_raw_shared_control_has_identical_capacity_and_shared_route(self):
        y=examples(6,72)
        models=[]
        for variant in ('joint_shared_gate','joint_raw_shared_gate'):
            torch.manual_seed(11)
            model=build_balanced_model(fit_statistics(y),joint_candidate(),variant)
            model.eval()
            features=cached(model,y,torch.device('cpu'))
            z3,z6,g=model.encode_tasks(features)
            torch.testing.assert_close(z3,z6)
            torch.testing.assert_close(g['task3'],g['task6'])
            models.append(model)
        self.assertEqual(sum(p.numel() for p in models[0].parameters()),sum(p.numel() for p in models[1].parameters()))
        for a,b in zip(models[0].parameters(),models[1].parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(models[1].arm,'geometry_moe')

    def test_shared_gate_fits_both_tasks_with_radius_unit_loss(self):
        y=examples(32,66)
        measure=np.linalg.norm(y[:,0,-1],axis=-1)
        labels=(measure>np.median(measure)).astype(np.int64)
        candidate=dict(joint_candidate(),classification_weight=1.)
        training=dict(updates=650,batch_size=16,warmup_updates=20,probe_every=325,gradient_clip=1.)
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            model,fit=train_balanced(y,labels,y,labels,'joint_shared_gate',candidate,training,7,
                                     torch.device('cpu'),Path(temp)/'fit')
            self.assertGreater(fit['train']['task3']['average_precision'],.98)
            self.assertLess(fit['train']['task6_rmse_r'],fit['curve'][0]['train']['task6_rmse_r']*.25)
            self.assertEqual(fit['train']['gate3_mean'],fit['train']['gate6_mean'])
            best=min(fit['curve'][1:],key=lambda row:row['validation']['selection_score'])
            self.assertEqual(fit['selected_step'],best['step'])
            self.assertIsNotNone(model.router[-1].weight.grad)
            self.assertFalse(list(Path(temp).rglob('*.pt')))

    def test_expansion_selection_and_all_final_controls_even_when_flagged(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            base=Path(temp)
            dataset='fixture'
            source,subset,root=[base/n for n in ('source','subset','result')]
            src,sub=source/'data'/dataset,subset/'data'/dataset
            for path in (src,sub,root):
                path.mkdir(parents=True)
            for path in (source/'config.frozen.json',subset/'config.frozen.json'):
                write_json(path,dict(fixture=True,path=str(path)))
            fieldpath=base/'flow.nc'
            with nc.Dataset(fieldpath,'w') as data:
                for name,values in [('t',np.arange(6)),('x',np.linspace(-1,1,21)),('y',np.linspace(-1,1,5)),('z',np.linspace(-1,1,5))]:
                    data.createDimension(name,len(values))
                    data.createVariable(name,'f8',(name,))[:]=values
                z,y,x=np.meshgrid(np.linspace(-1,1,5),np.linspace(-1,1,5),np.linspace(-1,1,21),indexing='ij')
                for name,value in [('u',x*0),('v',x*x),('w',z*0)]:
                    data.createVariable(name,'f4',('t','z','y','x'))[:]=np.broadcast_to(value,(6,5,5,21))
            files={}
            for i,role in enumerate(('train','validation','test','unseen_scale')):
                y=examples(48 if role=='train' else 12,100+i)
                start=0 if role=='train' else 2 if role=='validation' else 4
                origins=np.zeros((len(y),3))
                origins[:,0]=np.arange(len(y))%2
                np.savez(src/f'{role}.npz',geometry=y,scale_id=np.arange(len(y))%3,origin=origins,
                    seed_time=np.full(len(y),start),source_start=np.full(len(y),start),
                    primitive_id=np.arange(len(y))+i*100)
                files[role]=dict(sha256=sha256(src/f'{role}.npz'))
            write_json(src/'manifest.json',dict(files=files,source=dict(source=str(fieldpath),
                source_bytes=fieldpath.stat().st_size,source_mtime_ns=fieldpath.stat().st_mtime_ns)))
            write_json(src/'audit.json',dict(passed=True))
            small={}
            for seed in (91,92):
                ids=np.random.default_rng(seed).permutation(48)[:32]
                with np.load(src/'train.npz') as data:
                    np.savez(sub/f'train_{seed}.npz',geometry=data['geometry'][ids],source_indices=ids)
                small[f'train_{seed}.npz']=sha256(sub/f'train_{seed}.npz')
            with np.load(src/'validation.npz') as data:
                np.savez(sub/'validation.npz',geometry=data['geometry'])
            small['validation.npz']=sha256(sub/'validation.npz')
            write_json(sub/'manifest.json',dict(files=small,provenance=dict(config_sha256=sha256(subset/'config.frozen.json')),
                source_manifest_sha256=sha256(src/'manifest.json'),source_train_sha256=files['train']['sha256'],
                source_validation_sha256=files['validation']['sha256']))
            spec=dict(experiment='fixture',output_root=str(root),source_cache=str(source),subset_cache=str(subset),
                source_config_sha256=sha256(source/'config.frozen.json'),subset_config_sha256=sha256(subset/'config.frozen.json'),
                primary_train_size=24,datasets=[dataset],search_seed=91,seeds=[92],variants=list(VARIANTS),
                label_max_spatial_dim=96,candidate=joint_candidate(),validation_gate_rmse_r=3.,
                training=dict(updates=80,batch_size=8,warmup_updates=5,probe_every=40,gradient_clip=1.))
            config=base/'config.json'
            write_json(config,spec)
            (root/'config.frozen.json').write_bytes(config.read_bytes())
            # Generate the unchanged previous cache for the first 24 samples.
            previous.prepare(spec,config,0)
            old_spec,old_config=dict(spec),config
            spec=dict(spec,experiment='balanced_fixture',output_root=str(base/'balanced'),
                primary_train_size=40,train_sizes=[24,40],previous_cache=str(root),
                variants=list(VARIANTS)+['joint_raw_shared_gate'],
                previous_config_sha256=sha256(old_config),
                candidates=[dict(spec['candidate'],id=f'b{w}',classification_weight=float(w)) for w in (1,4)],
                selection='validation task metrics only',evaluation_policy='flag finite failures and test them',
                validation_gate_rmse_r=0.,training=dict(spec['training'],updates=16,probe_every=8))
            spec.pop('candidate')
            root=Path(spec['output_root'])
            root.mkdir()
            config=base/'balanced_config.json'
            write_json(config,spec)
            (root/'config.frozen.json').write_bytes(config.read_bytes())
            with patch.object(run,'gpu',return_value=torch.device('cpu')), \
                 patch.object(run,'load_truth',side_effect=AssertionError('Premature test read')):
                run.prepare(spec,config,0)
                for seed in (91,92):
                    with np.load(root/'data'/dataset/f'train_{seed}.npz') as new, np.load(sub/f'train_{seed}.npz') as old:
                        self.assertEqual(len(new['geometry']),40)
                        np.testing.assert_array_equal(new['geometry'][:32],old['geometry'])
                    labels=load_labels(spec,config,dataset,f'train_{seed}')
                    np.testing.assert_array_equal(labels[:24],load_labels(old_spec,old_config,dataset,f'train_{seed}'))
                self.assertFalse((root/'labels'/dataset/'test.npz').exists())
                for index in range(len(run.search_plan(spec))):
                    run.search(spec,config,index)
                run.select(spec,config)
            selection=run.frozen_selection(spec,config)
            self.assertTrue(selection['engineering_warnings'])
            self.assertEqual(set(selection['choices']),{'24','40'})
            run.prepare_test(spec,config,0)
            with patch.object(run,'gpu',return_value=torch.device('cpu')):
                for index in range(len(run.final_plan(spec))):
                    run.final(spec,config,index)
            for index in range(2):
                run.audit(spec,config,index)
            run.merge(spec,config)
            result=run.read(root/'final_audit.json')
            self.assertTrue(result['audit_passed'])
            self.assertEqual(len(result['summary']),32)
            self.assertTrue(all(row['complete'] for row in result['summary']))
            self.assertEqual(len(result['engineering_failures']),14)
            path=root/'final'/dataset/'24'/'92'/'joint_shared_gate'/'test_predictions.npz'
            with np.load(path) as cache:
                arrays={k:cache[k].copy() for k in cache.files}
            arrays['prediction'][0,0,1,0]+=1
            np.savez(path,**arrays)
            with self.assertRaises(AssertionError):
                run.audit(spec,config,0)
            self.assertFalse(list(root.rglob('*.pt')))


if __name__=='__main__':
    unittest.main()
