import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments import Task4C_FMTOptimization_1_1 as run


class OptimizationTests(unittest.TestCase):
    def test_diagnostic_loop_preserves_original_optimizer_updates(self):
        torch.manual_seed(96611)
        template = torch.nn.Sequential(torch.nn.Linear(7, 12), torch.nn.GELU(), torch.nn.Dropout(.15), torch.nn.Linear(12, 2))
        d = SimpleNamespace(clean=torch.randn(73, 7), targets=torch.arange(73) % 2)
        order = np.random.default_rng(96611).permutation(73); states = []; losses = []
        for instrumented in (False, True):
            model = copy.deepcopy(template); opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
            torch.manual_seed(96611)
            if instrumented:
                loss, report = run.fit_epoch(model, d, opt, order, 16, True)
                self.assertTrue(np.isfinite(list(report.values())).all())
                self.assertGreaterEqual(report['clipped_step_fraction'], 0)
                self.assertLessEqual(report['clipped_step_fraction'], 1)
            else:
                model.train(); total = 0.
                for first in range(0, len(order), 16):
                    ids = order[first:first+16]; opt.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(d.clean[ids]), d.targets[ids])
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); opt.step()
                    total += float(loss.detach())*len(ids)
                loss = total/len(order)
            states.append(model.state_dict()); losses.append(loss)
        self.assertEqual(losses[0], losses[1])
        for key in states[0]: torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)

    def test_dropout_override_preserves_weights_and_parameter_count(self):
        for name in ('p35_h0', 'c156'):
            torch.manual_seed(96611); model = run.base.old.fmt_old.baseline.method.make_model(name)
            state = copy.deepcopy(model.state_dict())
            with patch.object(run.base, 'make_model', return_value=model):
                changed = run.make_model(dict(base_method=name), dict(dropout_override=.35))
            drop = [m.p for m in changed.modules() if isinstance(m, torch.nn.Dropout)]
            self.assertTrue(drop); self.assertTrue(all(p == .35 for p in drop))
            for key in state: torch.testing.assert_close(state[key], changed.state_dict()[key], atol=0, rtol=0)

    def test_data_labels_lengths_and_neighbor_protocol_remain_frozen(self):
        s = run.spec(); old = json.loads(Path(run.base.CONFIG).read_text())
        for key in ('source_output', 'source_audit_sha256', 'neighbor_output', 'flows', 'folds', 'expected_counts', 'selection', 'training'):
            self.assertEqual(s[key], old[key])
        self.assertEqual(s['candidates'], old['candidates'][:4]); self.assertEqual(s['seed'], 96611)
        self.assertEqual(s['screen'], dict(epochs=12, patience=None)); self.assertEqual(len(s['recipes'])*len(s['candidates']), 24)
        self.assertNotIn('validation', s)

    def test_rank_uses_f1_then_ap_and_retains_first_exact_tie(self):
        h = [dict(epoch=i+1, test=dict(combined=dict(f1=f, average_precision=a)))
             for i,(f,a) in enumerate([(0.6,.5),(.59,.9),(.6,.6),(.6,.6)])]
        self.assertEqual(run.update_rank(h), (.6,.6,3))


if __name__ == '__main__': unittest.main()
