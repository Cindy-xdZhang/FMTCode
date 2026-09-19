import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from FMT_Utils import Task4C_V2NewLabelTraining_1_1 as data
from experiments import Task4C_V2NewLabelTraining_1_1 as run


class NewLabelTrainingTests(unittest.TestCase):
    def test_four_full_training_folds_no_validation_and_no_split_randomness(self):
        fold = np.repeat(np.arange(5), [7, 3, 11, 9, 5])
        s = run.spec()
        a = data.split_masks(dict(fold=fold), 0, s)
        b = data.split_masks(dict(fold=fold), 1, s)
        self.assertEqual(set(a), {'train', 'test'})
        np.testing.assert_array_equal(np.flatnonzero(a['train']), np.arange(7, 35))
        for role in a: np.testing.assert_array_equal(a[role], b[role])
        self.assertFalse(np.any(a['train'] & a['test']))
        self.assertTrue(np.all(a['train'] | a['test']))

    def test_actual_reader_uses_new_labels_all_lengths_and_original_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rng = np.random.default_rng(42); s = copy.deepcopy(run.spec())
            s.update(source_output=str(root/'source'), neighbor_output=str(root/'neighbors'), run_folder=str(root/'run'))
            s['expected_counts'] = dict(train=192, test=48)
            for fi, f in enumerate(s['flows']):
                name = f['name']; src = root/'source'/'physical'/name; src.mkdir(parents=True)
                nei = root/'neighbors'/name; nei.mkdir(parents=True)
                seeds = rng.normal(size=(40, 3)); curves = rng.normal(size=(40, 3, 32, 3)).astype(np.float32)
                np.save(src/'seeds.npy', seeds); np.save(src/'curves.npy', curves)
                labels = np.arange(40) % 2; folds = np.arange(40) % 5
                np.savez(src/'metadata.npz', label=labels, seed_label=1-labels, instance=folds,
                    fold=folds, assignment_kind=np.zeros(40, np.int8))
                for k in (6, 16):
                    order = (np.arange(40)[:, None]+np.arange(1, k+1)[None]) % 40
                    np.save(nei/f'order{k}.npy', order)
            for k in (6, 16):
                c = dict(neighbors=k)
                train = data.Dataset(s, 'train', c, device='cpu'); test = data.Dataset(s, 'test', c, device='cpu')
                for role, d in [('train', train), ('test', test)]:
                    self.assertEqual(len(d.labels), s['expected_counts'][role])
                    for fi in range(2):
                        take = d.flows == fi
                        ids = np.flatnonzero(folds != 0) if role == 'train' else np.flatnonzero(folds == 0)
                        np.testing.assert_array_equal(d.rows[take], np.repeat(ids, 3))
                        np.testing.assert_array_equal(d.lengths[take], np.tile([0, 1, 2], len(ids)))
                        np.testing.assert_array_equal(d.labels[take], np.repeat(labels[ids], 3))
                    rows = np.array([len(d.labels)-1, 0, 7, len(d.labels)//2, 7])
                    actual, counts = d.geometry(rows)
                    for j, row in enumerate(rows):
                        source = d.parts[int(d.flows[row])][0]
                        lines, ss, _ = source.gather(np.array([d.rows[row]]), np.array([d.lengths[row]]))
                        expected, _, _ = data.old.v2.normalize_bundle(torch.tensor(lines), torch.tensor(ss, dtype=torch.float32))
                        np.testing.assert_array_equal(actual[j, :k+1], expected[0].numpy())
                        self.assertTrue((actual[j, k+1:] == 0).all()); self.assertEqual(counts[j], k+1)
                with self.assertRaises(AssertionError): data.Dataset(s, 'validation', c, device='cpu')
                for dataset in (train, test):
                    for source, _, _ in dataset.parts:
                        for array in (source.seeds, source.curves, source.neighbors): array._mmap.close()

    def test_frozen_data_counts_and_six_single_seed_methods(self):
        s = run.spec(); n = json.loads(Path(run.NEIGHBOR_CONFIG).read_text())
        self.assertEqual(s['final'], dict(seeds=[96611], epochs=500, patience=50))
        self.assertEqual([(c['base_method'], c['neighbors']) for c in s['candidates']],
            [('p35_h0', 6), ('p35_h0', 16), ('c156', 6), ('c156', 16), ('conv16', 6), ('conv16', 16)])
        for key in ('source_output', 'source_audit_sha256', 'flows', 'folds', 'expected_counts'):
            self.assertEqual(s[key], n[key])
        old = json.loads(Path('config/Ablation_Task4C_BallQuery_2.5.json').read_text())
        self.assertEqual(s['training'], old['training']); self.assertEqual(s['candidates'][:4], old['candidates'])
        self.assertEqual(s['selection']['role'], 'test'); self.assertFalse(s['selection']['test_labels_in_gradients'])


if __name__ == '__main__': unittest.main()
