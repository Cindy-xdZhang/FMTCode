"""Exercise nested expansion, fresh controls, selection and final evaluation."""
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256, write_json
from experiments import Task6_PNNDataSchedule_1_3 as run
from experiments.Report_Task6_PNNDataSchedule_1_3 import report
from tests.test_task6_direct_neural_5_1 import examples
from tests.test_task6_pnn_tuning_1_2 import candidate


class TestExpandedPipeline(unittest.TestCase):
    def test_nested_expansion_fresh_baseline_and_test_gate(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            base = Path(temp)
            dataset = 'unit_fixture'
            source, subset, previous, root = [base/name for name in ('source','subset','previous','result')]
            src, sub = source/'data'/dataset, subset/'data'/dataset
            for path in (src, sub, previous, root):
                path.mkdir(parents=True)
            for path in (source/'config.frozen.json',subset/'config.frozen.json',previous/'config.frozen.json'):
                write_json(path,dict(unit_fixture=True,path=str(path)))
            files = {}
            for i, role in enumerate(('train','validation','test','unseen_scale')):
                y = examples(48 if role == 'train' else 8,100+i)
                np.savez(src/f'{role}.npz',geometry=y,scale_id=np.arange(len(y))%2)
                files[role] = dict(sha256=sha256(src/f'{role}.npz'))
            write_json(src/'manifest.json',dict(files=files))
            write_json(src/'audit.json',dict(passed=True))
            small_files = {}
            for seed in (91,92):
                ids = np.random.default_rng(seed).permutation(48)[:32]
                with np.load(src/'train.npz') as data:
                    np.savez(sub/f'train_{seed}.npz',geometry=data['geometry'][ids],source_indices=ids)
                small_files[f'train_{seed}.npz'] = sha256(sub/f'train_{seed}.npz')
                if seed == 91:
                    prefix_hash = hashlib.sha256(ids[:8].tobytes()).hexdigest()
            with np.load(src/'validation.npz') as data:
                np.savez(sub/'validation.npz',geometry=data['geometry'])
            small_files['validation.npz'] = sha256(sub/'validation.npz')
            write_json(sub/'manifest.json',dict(files=small_files,
                provenance=dict(config_sha256=sha256(subset/'config.frozen.json')),
                source_manifest_sha256=sha256(src/'manifest.json'),source_train_sha256=files['train']['sha256'],
                source_validation_sha256=files['validation']['sha256']))
            c = candidate()
            old = dict(c)
            c['scheduler'] = dict(name='one_cycle',pct_start=.1,div_factor=25.,final_div_factor=400.)
            for name in ('selection.json','selection.before_test.json'):
                write_json(previous/name,dict(test_read=False,candidate=old))
            old_result = previous/'refine'/dataset/old['id']/'result.json'
            old_result.parent.mkdir(parents=True)
            write_json(old_result,dict(provenance=dict(config_sha256=sha256(previous/'config.frozen.json')),
                                      subset_sha256=prefix_hash))
            training = dict(updates=100,batch_size=8,warmup_updates=10,probe_every=25,gradient_clip=1.)
            spec = dict(experiment='unit_fixture',output_root=str(root),source_cache=str(source),subset_cache=str(subset),
                source_config_sha256=sha256(source/'config.frozen.json'),subset_config_sha256=sha256(subset/'config.frozen.json'),
                previous_cache=str(previous),previous_config_sha256=sha256(previous/'config.frozen.json'),
                previous_selection_sha256=sha256(previous/'selection.json'),previous_candidate=old['id'],
                reference_candidate=c['id'],datasets=[dataset],screen_datasets=[dataset],
                arms=['pnn_trans','raw_trans','raw_frozen'],train_sizes=[24],primary_train_size=24,previous_train_size=8,
                search_seed=91,seeds=[92],training=training,screening=training,candidates=[c],promote_count=1,
                validation_gate_rmse_r=3.,selection='fixture',
                baseline_candidate=dict(id='c1',learning_rate=.003,dropout=0.,weight_decay=0.),
                baseline_training=dict(training,architecture=c['architecture']))
            config = base/'config.json'
            write_json(config,spec)
            (root/'config.frozen.json').write_bytes(config.read_bytes())
            with patch.object(run,'gpu',return_value=torch.device('cpu')), \
                 patch('torch.cuda.get_device_name',return_value='CPU fixture'), \
                 patch.object(run,'evaluate',side_effect=AssertionError('Premature test read')):
                run.prepare(spec,config)
                run.baseline(spec,config,0)
                run.screen(spec,config,0)
                run.promote(spec,config)
                run.refine(spec,config,0)
                run.select(spec,config)
            self.assertEqual(run.read(root/'data_expansion.json')['expansion_factor'],3)
            self.assertEqual(run.baseline_rows(spec)[0]['train_size'],24)
            selection = run.read(root/'selection.json')
            checked = report(root,config)
            self.assertEqual(checked['baseline_models'],1)
            self.assertEqual(checked['screen_models'],1)
            self.assertEqual(checked['refine_models'],2)
            self.assertFalse(selection['test_read'])
            self.assertEqual(selection['candidate']['id'],c['id'])
            negative = dict(selection,gate_passed=False)
            for name in ('selection.json','selection.before_test.json'):
                write_json(root/name,negative)
            with self.assertRaises(AssertionError):
                run.final(spec,config,0)
            # Positive gate fixture exercises actual final training independently of toy ranking.
            positive = dict(selection,gate_passed=True,unit_test_gate_fixture=True)
            for name in ('selection.json','selection.before_test.json'):
                write_json(root/name,positive)
            with patch.object(run,'gpu',return_value=torch.device('cpu')), \
                 patch('torch.cuda.get_device_name',return_value='CPU fixture'):
                run.final(spec,config,0)
            result = run.read(root/'final'/dataset/'24'/'92'/'result.json')
            self.assertEqual(sorted(r['arm'] for r in result['fits']),['pnn_trans','raw_frozen','raw_trans'])
            run.audit(spec,config,0)
            run.merge(spec,config)
            self.assertTrue(run.read(root/'final_audit.json')['audit_passed'])
            self.assertFalse(list(root.rglob('*.pt')))


if __name__ == '__main__':
    unittest.main()
