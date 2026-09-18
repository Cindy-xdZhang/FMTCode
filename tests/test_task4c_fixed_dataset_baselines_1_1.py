"""Family specs and phase plans of the fixed-dataset baseline runs 1.1."""
import json
from pathlib import Path
import unittest
from experiments import Task4C_FixedDatasetBaselines_1_1 as runner


class FixedDatasetBaselineTests(unittest.TestCase):
    def test_every_family_spec_inherits_frozen_training_rules_and_new_data_keys(self):
        base = json.loads(Path('config/Ablation_Task4C_BottomDensity_1.2.json').read_text())
        for family, module in runner.MODULES.items():
            spec = runner.load_spec(runner.CONFIG, family)
            self.assertEqual({k: spec['training'][k] for k in base['training'] if k != 'seeds'}, {k: base['training'][k] for k in base['training'] if k != 'seeds'})
            self.assertEqual(spec['training']['seeds'], [96611, 96612, 96613]); self.assertEqual(spec['encoding']['fmt_definition'], base['encoding']['fmt_definition'])
            self.assertEqual(spec['expected_counts'], dict(train=180000, validation=3000, test=12000, total=195000))
            self.assertIn('FMT_Task4C_FixedDataset_1p1', spec['source_output']); self.assertTrue(spec['output'].endswith(family))
            for phase, array, gpu, wall in runner.PLANS[family]:
                if phase not in ('reuse', 'summarize'): self.assertTrue(hasattr(module, phase), (family, phase))
        self.assertEqual(runner.load_spec(runner.CONFIG, 'conv')['methods'], ['p35', 'conv16', 'conv24'])
        self.assertEqual(runner.load_spec(runner.CONFIG, 'conv')['encoding']['voxel_resolutions'], [16, 24])
        self.assertEqual(runner.load_spec(runner.CONFIG, 'pointnetpp')['pointnetplusplus']['parameters'], 76723)
        self.assertEqual(runner.load_spec(runner.CONFIG, 'pointnet')['pointnet']['parameters'], 76749)
        self.assertEqual(runner.load_spec(runner.CONFIG, 'bilstm')['baseline_definitions']['baseline2']['parameters'], 76786)

    def test_rebinding_keeps_frozen_code_but_this_identity(self):
        import inspect
        fn = runner.engine.train; rebound = runner.FunctionType(fn.__code__, dict(runner.engine.__dict__, identity=runner.identity), argdefs=fn.__defaults__)
        self.assertIs(rebound.__globals__['identity'], runner.identity); self.assertEqual(inspect.signature(rebound), inspect.signature(fn))


if __name__ == '__main__': unittest.main()
