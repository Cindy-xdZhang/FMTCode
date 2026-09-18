"""Arm/seed indexing and config contracts of the fixed-dataset FMT comparison 1.1."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from experiments import Task4C_FixedDatasetFMT_1_1 as runner
from FMT_Utils import Task4C_FPS16_1_1 as method


class FixedDatasetFMTTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(Path(runner.CONFIG).read_text()); self.fps16 = json.loads(Path('config/Ablation_Task4C_FPS16_1.2.json').read_text())

    def test_arm_spec_enumerates_every_candidate_and_seed_once(self):
        seeds = self.spec['final']['seeds']; seen = set()
        for index in range(len(self.spec['candidates'])*len(seeds)):
            arm = runner.arm_spec(self.spec, index); seen.add((arm['candidate']['id'], arm['_active_seed']))
            self.assertEqual(arm['final']['seeds'], [arm['_active_seed']]); self.assertEqual(Path(arm['output']).parts[-2:], (arm['candidate']['id'], f"seed{arm['_active_seed']}"))
        self.assertEqual(len(seen), len(self.spec['candidates'])*len(seeds))

    def test_config_keeps_frozen_arms_and_three_seeds_on_the_fixed_dataset(self):
        self.assertEqual(self.spec['candidates'], self.fps16['candidates']); self.assertEqual(self.spec['final']['seeds'], [96611, 96612, 96613])
        self.assertEqual(self.spec['expected_counts'], dict(train=180000, validation=3000, test=12000))
        self.assertIn('FMT_Task4C_FixedDataset_1p1', self.spec['source_output']); self.assertEqual(self.spec['training'], self.fps16['training'])

    def test_slot0_center_encoding_matches_frozen_fps16_encoder(self):
        torch.manual_seed(5); g = torch.randn(3, 27, 32, 3); s = torch.randn(3, 27, 3); c = torch.tensor([17, 21, 27]); a = torch.zeros(3, dtype=torch.long)
        for i, n in enumerate(c): g[i, n:] = 0; s[i, n:] = 0
        for cand in self.spec['candidates']:
            x, n = method.encode(g, s, c, a, cand); self.assertEqual(list(x.shape), [3, 1, 142]); self.assertTrue(torch.isfinite(x).all())
            self.assertEqual(n.shape, (3, cand['neighbors'])); self.assertTrue(bool((n != 0).all()))
            model = method.make_model(cand['base_method']); self.assertEqual(sum(p.numel() for p in model.parameters()), cand['parameters'])


if __name__ == '__main__': unittest.main()
