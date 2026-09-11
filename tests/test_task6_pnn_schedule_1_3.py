"""Schedule semantics and a real CPU training path, without scientific claims."""
import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from FMT_Utils.Task6PNNSchedule_3D import LearningRateSchedule, train_scheduled
from FMT_Utils.Task6PNNTuning_3D import train_tuned
from experiments.Task6_PNNDataSchedule_1_3 import promotion_ranking, selection_result
from tests.test_task6_pnn_tuning_1_2 import candidate
from tests.test_task6_direct_neural_5_1 import examples


class TestSchedules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def schedule(self, policy, updates=100, warmup=10):
        parameter = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.AdamW([parameter], lr=.001)
        schedule = LearningRateSchedule(optimizer, dict(learning_rate=.001, scheduler=policy),
                                       dict(updates=updates, warmup_updates=warmup))
        return optimizer, schedule

    def test_cosine_identical_to_frozen_formula_and_training(self):
        opt, schedule = self.schedule(dict(name='cosine_legacy'))
        for step in range(100):
            expected = .001*min(1.,(step+1)/10)*(.1+.45*(1+math.cos(math.pi*step/99)))
            self.assertEqual(schedule.before_update(step), expected)
            opt.step()
            schedule.after_update()
        y = examples(24, 37)
        c = candidate()
        c['scheduler'] = dict(name='cosine_legacy')
        training = dict(updates=30,batch_size=8,warmup_updates=4,probe_every=15,gradient_clip=1.)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            old_model, old = train_tuned(y,y,'pnn_trans',c,training,91,torch.device('cpu'),Path(temp)/'old')
            new_model, new = train_scheduled(y,y,'pnn_trans',c,training,91,torch.device('cpu'),Path(temp)/'new')
            for a,b in zip(old['curve'],new['curve']):
                self.assertEqual(a['validation_rmse_r'], b['validation_rmse_r'])
            for key,value in old_model.state_dict().items():
                torch.testing.assert_close(value,new_model.state_dict()[key],rtol=0,atol=0)

    def test_onecycle_endpoints_peak_and_optimizer_order(self):
        opt, schedule = self.schedule(dict(name='one_cycle',pct_start=.1,div_factor=25.,final_div_factor=400.))
        rates = []
        for step in range(100):
            rates.append(schedule.before_update(step))
            opt.step()
            schedule.after_update()
        self.assertAlmostEqual(rates[0],.001/25,places=12)
        self.assertAlmostEqual(max(rates),.001,places=12)
        self.assertAlmostEqual(rates[-1],.001/10000,places=12)
        self.assertEqual(np.argmax(rates),9)
        self.assertEqual(opt.param_groups[0]['betas'],(.9,.999))
        self.assertEqual(schedule.completed,100)

    def test_plateau_ignores_initial_probe_then_reduces_after_three_bad(self):
        policy = dict(name='plateau',factor=.5,patience=2,threshold=.001,min_lr_ratio=.01)
        opt, schedule = self.schedule(policy, updates=12, warmup=2)
        schedule.observe_validation(0, 0.)
        values = {2:1.,4:1.,6:1.,8:1.,10:.8,12:.8}
        rates = []
        for step in range(12):
            rates.append(schedule.before_update(step))
            opt.step()
            schedule.after_update()
            if step+1 in values:
                schedule.observe_validation(step+1, values[step+1])
        self.assertEqual(rates[0],.0005)
        self.assertEqual(rates[1:8],[.001]*7)
        self.assertEqual(rates[8:],[.0005]*4)
        self.assertEqual(schedule.validation_steps,[2,4,6,8,10,12])
        with self.assertRaises(ValueError):
            schedule.observe_validation(12,.8)

    def test_all_policies_train_and_preserve_latent(self):
        y = examples(24, 3)
        policies = [dict(name='cosine_legacy'),
                    dict(name='one_cycle',pct_start=.1,div_factor=25.,final_div_factor=400.),
                    dict(name='plateau',factor=.5,patience=2,threshold=.001,min_lr_ratio=.01)]
        training = dict(updates=150,batch_size=8,warmup_updates=10,probe_every=30,gradient_clip=1.)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            for policy in policies:
                c = candidate()
                c['scheduler'] = policy
                _, result = train_scheduled(y,y,'pnn_trans',c,training,17,torch.device('cpu'),Path(temp)/policy['name'])
                self.assertLess(result['train_rmse_r'],result['curve'][0]['train_rmse_r']*.5)
                self.assertGreater(result['train_zero_latent_rmse_r'],result['train_rmse_r'])
                self.assertEqual(result['train_examples_exposed'],150*8)
                self.assertEqual(result['schedule']['completed_updates'],150)
                self.assertTrue(all(r['learning_rate_used']>0 for r in result['curve'][1:]))
            self.assertFalse(list(Path(temp).rglob('*.pt')))

    def test_reference_retained_and_frozen_control_in_numeric_gate(self):
        spec = dict(candidates=[dict(id=str(i)) for i in range(5)], screen_datasets=['a','b'],
                    datasets=['a','b'],promote_count=4,reference_candidate='4',validation_gate_rmse_r=3.,selection='fixture')
        baseline = [dict(dataset=d,validation_rmse_r=1.) for d in spec['datasets']]
        screen = [dict(dataset=d,candidate=str(i),validation_rmse_r=.1*(i+1))
                  for d in spec['datasets'] for i in range(5)]
        _, chosen = promotion_ranking(screen,baseline,spec)
        self.assertEqual([c['id'] for c in chosen],['0','1','2','4'])
        rows = [dict(dataset=d,candidate=c['id'],arm=arm,validation_rmse_r=v)
                for d in spec['datasets'] for c in chosen for arm,v in [('pnn_trans',.5),('raw_trans',.6)]]
        self.assertTrue(selection_result(rows,baseline,chosen,spec)['gate_passed'])
        baseline[0]['validation_rmse_r'] = 3.1
        self.assertFalse(selection_result(rows,baseline,chosen,spec)['gate_passed'])


if __name__ == '__main__':
    unittest.main()
