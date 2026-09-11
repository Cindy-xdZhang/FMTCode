"""Run real fitting through the full pipeline and reject damaged evaluations."""
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256,write_json
from experiments import Task6_FMTGeometryMoE_1_1 as run
from experiments.Report_Task6_FMTGeometryMoE_1_1 import report
from tests.test_task6_direct_neural_5_1 import examples
from tests.test_task6_fmt_geometry_moe_1_1 import candidate
from tests.test_task6_pnn_tuning_1_2 import candidate as raw_candidate


class TestMoEPipeline(unittest.TestCase):
    def test_complete_pipeline_and_independent_audit(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            base=Path(temp)
            dataset='unit_fixture'
            source,subset,previous,root=[base/n for n in ('source','subset','previous','result')]
            src,sub=source/'data'/dataset,subset/'data'/dataset
            for path in (src,sub,previous,root):
                path.mkdir(parents=True)
            for path in (source/'config.frozen.json',subset/'config.frozen.json',previous/'config.frozen.json'):
                write_json(path,dict(unit_fixture=True,path=str(path)))
            files={}
            for i,role in enumerate(('train','validation','test','unseen_scale')):
                y=examples(48 if role=='train' else 8,100+i)
                np.savez(src/f'{role}.npz',geometry=y,scale_id=np.arange(len(y))%2)
                files[role]=dict(sha256=sha256(src/f'{role}.npz'))
            write_json(src/'manifest.json',dict(files=files))
            write_json(src/'audit.json',dict(passed=True))
            small_files={}
            for seed in (91,92):
                ids=np.random.default_rng(seed).permutation(48)[:32]
                with np.load(src/'train.npz') as data:
                    np.savez(sub/f'train_{seed}.npz',geometry=data['geometry'][ids],source_indices=ids)
                small_files[f'train_{seed}.npz']=sha256(sub/f'train_{seed}.npz')
                if seed==91:
                    prefix_hash=hashlib.sha256(ids[:24].tobytes()).hexdigest()
            with np.load(src/'validation.npz') as data:
                np.savez(sub/'validation.npz',geometry=data['geometry'])
            small_files['validation.npz']=sha256(sub/'validation.npz')
            write_json(sub/'manifest.json',dict(files=small_files,
                provenance=dict(config_sha256=sha256(subset/'config.frozen.json')),
                source_manifest_sha256=sha256(src/'manifest.json'),source_train_sha256=files['train']['sha256'],
                source_validation_sha256=files['validation']['sha256']))
            for name in ('selection.json','selection.before_test.json'):
                write_json(previous/name,dict(test_read=False,fixture=True))
            baseline=previous/'baseline'/dataset/'result.json'
            baseline.parent.mkdir(parents=True)
            write_json(baseline,dict(dataset=dataset,train_size=24,subset_sha256=prefix_hash,test_read=False,
                validation_rmse_r=.1,provenance=dict(config_sha256=sha256(previous/'config.frozen.json'))))
            training=dict(updates=100,batch_size=8,warmup_updates=10,probe_every=50,gradient_clip=1.)
            spec=dict(experiment='unit_fixture',output_root=str(root),source_cache=str(source),subset_cache=str(subset),
                source_config_sha256=sha256(source/'config.frozen.json'),subset_config_sha256=sha256(subset/'config.frozen.json'),
                baseline_cache=str(previous),baseline_config_sha256=sha256(previous/'config.frozen.json'),
                baseline_selection_sha256=sha256(previous/'selection.json'),
                datasets=[dataset],screen_datasets=[dataset],families=['raw_mlp','pointnn'],
                arms=['fmt_moe','geometry_moe','geometry_only'],primary_train_size=24,search_seed=91,seeds=[92],
                latent_dimension=12,
                training=training,screening=training,candidates=[candidate(),candidate('pointnn','channel')],
                promote_per_family=1,validation_gate_rmse_r=3.,selection='fixture',
                fit_check=dict(training,samples=8,updates=350,probe_every=175,
                    maximum_error_ratio=.3,minimum_zero_latent_error_ratio=2.),
                baseline_candidate=dict(id='c1',learning_rate=.003,dropout=0.,weight_decay=0.),
                baseline_training=dict(training,architecture=raw_candidate()['architecture']))
            config=base/'config.json'
            write_json(config,spec)
            (root/'config.frozen.json').write_bytes(config.read_bytes())
            with patch.object(run,'gpu',return_value=torch.device('cpu')), \
                 patch('torch.cuda.get_device_name',return_value='CPU fixture'), \
                 patch.object(run,'evaluate',side_effect=AssertionError('Premature test read')):
                run.prepare(spec,config)
                run.fit_check(spec,config,0)
                for index in range(2):
                    run.screen(spec,config,index)
                run.promote(spec,config)
                for index in range(2):
                    run.refine(spec,config,index)
                run.select(spec,config)
            selection=run.read(root/'selection.json')
            checked=report(root,config)
            self.assertEqual(checked['completed_models'],dict(fit_check=2,screen=2,refine=6))
            self.assertFalse(selection['test_read'])
            self.assertTrue(selection['gate_passed'])
            self.assertTrue(all(r['qualified'] for r in selection['families'].values()))
            for name in ('selection.json','selection.before_test.json'):
                write_json(root/name,dict(selection,gate_passed=False))
            with self.assertRaises(AssertionError):
                run.final(spec,config,0)
            for name in ('selection.json','selection.before_test.json'):
                write_json(root/name,selection)
            with patch.object(run,'gpu',return_value=torch.device('cpu')), \
                 patch('torch.cuda.get_device_name',return_value='CPU fixture'):
                for index in range(3):
                    run.final(spec,config,index)
            run.audit(spec,config,0)
            run.merge(spec,config)
            result=run.read(root/'final_audit.json')
            self.assertTrue(result['audit_passed'])
            self.assertEqual(len(result['summary']),14)
            self.assertTrue(all(r['complete'] for r in result['summary']))
            self.assertFalse(result['failures'])
            checked=report(root,config)
            self.assertEqual(checked['completed_models']['final'],7)
            self.assertFalse(list(root.rglob('*.pt')))
            path=root/'final'/dataset/'92'/'raw_mlp'/'fmt_moe'/'test_predictions.npz'
            with np.load(path) as data:
                prediction=data['prediction'].copy()
            prediction[0,0,10,0]+=1
            np.savez(path,prediction=prediction)
            with self.assertRaises(AssertionError):
                run.audit(spec,config,0)


if __name__=='__main__':
    unittest.main()
