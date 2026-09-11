import contextlib
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import Record_Task6_DirectNeural_5_1 as record


class TestAutomaticRecord(unittest.TestCase):
    def test_macro_uses_equal_flow_seed_means_and_preserves_failure(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)
                root=Path('outputs/fixture')
                for folder in (root, Path('config'), Path('docs')):
                    folder.mkdir(parents=True)
                for name in ('experiment_log.md', 'ibex_run_registry.md'):
                    Path('docs',name).write_text('Historical results remain.\n')
                config=Path('config/Verify_Task6_DirectNeural_5.1.json')
                config.write_text(json.dumps(dict(output_root=str(root), train_sizes=[16],datasets=['a','b'],seeds=[1,2])))
                (root/'config.frozen.json').write_bytes(config.read_bytes())
                (root/'submissions.jsonl').write_text(json.dumps(dict(job_id='1',git_commit='unit_fixture'))+'\n')
                (root/'selection.json').write_text('{}')
                (root/'selection.before_test.json').write_text('{}')
                rows=[]
                for role in ('test','unseen_scale'):
                    for method, factor in [('fmt',1),('raw_matched',2),('raw_selected',1.5),('pca',.5)]:
                        for d, values in [('a',[1.,3.]),('b',[2.,6.])]:
                            for seed,value in enumerate(values,1):
                                rows.append(dict(train_size=16,dataset=d,seed=seed,role=role,
                                    comparison=method,scale_id=-1,position_rmse_r=value*factor))
                with (root/'metrics.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                audit=dict(passed=True,models=4,rows=len(rows),selection_sha256=record.digest(root/'selection.json'))
                (root/'final_audit.json').write_text(json.dumps(audit))
                with patch.object(record.subprocess,'check_output',return_value='1|1|t6-direct-merge|COMPLETED|s|a|b|node|0:0\n'):
                    record.main()
                    report=record.read(root/'automatic_end_record.json')
                    self.assertEqual(report['macro'][0]['fmt'],3.)
                    self.assertEqual(report['macro'][0]['raw_matched'],6.)
                    self.assertEqual(report['macro'][0]['relative_gain_vs_raw_matched_percent'],50.)
                    self.assertEqual(report['macro'][0]['fmt_wins_vs_raw_matched'],2)
                    text=Path('docs/experiment_log.md').read_text(encoding='utf-8')
                    record.main()
                    self.assertEqual(Path('docs/experiment_log.md').read_text(encoding='utf-8'),text)
                    (root/'final_audit.json').write_text(json.dumps(dict(passed=False,failures=['fixture_failure'])))
                    record.main()
                    self.assertIn('fixture_failure',Path('docs/experiment_log.md').read_text(encoding='utf-8'))
                    self.assertTrue(Path('docs/experiment_log.md').read_text().startswith('Historical results remain.'))
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
