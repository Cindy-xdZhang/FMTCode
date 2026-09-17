"""Independently recompute expanded/old/added GT-head results from predictions."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def metrics(y, p):
    predicted = p >= .5
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(f1=float(f1_score(y, predicted, zero_division=0)),
                precision=float(precision_score(y, predicted, zero_division=0)),
                recall=float(recall_score(y, predicted, zero_division=0)),
                average_precision=float(average_precision_score(y, p)),
                accuracy=float(np.mean(predicted == y)), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def audit(spec, config):
    root, source = Path(spec['output']), Path(spec['source_output'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    source_check = json.loads((root/'source_verification.json').read_text())
    assert source_check['complete'] and len(source_check['files']) == 18
    for item in source_check['files']:
        assert sha(source/'physical'/item['flow']/item['role']/item['file']) == item['sha256']
    results, per_gt, per_seed = [], [], []
    for seed in spec['final']['seeds']:
        folder = root/'final/p35_h0'/f'seed{seed}'
        record = json.loads((folder/'result.json').read_text())
        assert record['complete'] and record['parameters'] == 76738
        assert record['identity']['git_commit'] == commit and record['identity']['config_sha256'] == sha(config)
        lock = json.loads((folder/'selection.lock.json').read_text())
        assert lock['test_loaded'] is False and lock['threshold'] == .5
        best = max(record['history'], key=lambda h: (h['validation_f1'], h['validation_average_precision']))
        assert best['epoch'] == record['selected_epoch'] == lock['selected_epoch']
        independent_history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
        assert independent_history == record['history']
        row = dict(seed=seed, selected_epoch=best['epoch'], epochs=record['epochs'],
                   training_seconds=record['training_seconds'])
        for role in ('validation', 'test'):
            prediction = folder/f'{role}_predictions.npz'
            assert sha(prediction) == record['predictions'][role]
            with np.load(prediction) as z: values = {k: z[k] for k in z.files}
            y, p = values['labels'], values['probability']
            assert len(y) == spec['expected_counts'][role] and float(values['threshold']) == .5
            assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
            for fi, flow in enumerate(spec['flows']):
                sel = values['flow_index'] == fi; indices = values['row_in_split'][sel]
                with np.load(source/'physical'/flow['name']/role/'metadata.npz') as original:
                    assert np.array_equal(np.sort(indices), np.arange(len(original['labels'])))
                    assert np.array_equal(y[sel], original['labels'][indices])
                    assert np.array_equal(values['instance'][sel], original['instance'][indices])
            score = metrics(y, p)
            reference = record['validation'] if role == 'validation' else record['test']['combined']
            for key in ('f1', 'precision', 'recall', 'average_precision', 'accuracy', 'tp', 'fp', 'fn', 'tn'):
                if key in reference: assert abs(score[key]-reference[key]) < 1e-12, key
            row[role] = score
            if role == 'test':
                old = np.zeros(len(y), dtype=bool)
                row['per_flow'] = {}
                for fi, flow in enumerate(spec['flows']):
                    sel = values['flow_index'] == fi
                    old |= sel & (values['row_in_split'] < spec['old_test_per_flow'][flow['name']])
                    row['per_flow'][flow['name']] = metrics(y[sel], p[sel])
                assert old.sum() == 10000 and (~old).sum() == 1320 and np.all(y[~old] == 1)
                row['original_test'] = metrics(y[old], p[old])
                row['added_head'] = dict(samples=1320, correct=int(np.sum(p[~old] >= .5)),
                                         missed=int(np.sum(p[~old] < .5)), recall=float(np.mean(p[~old] >= .5)))
                for fi, flow in enumerate(spec['flows']):
                    sel = (~old) & (values['flow_index'] == fi)
                    instances = np.unique(values['instance'][sel])
                    assert len(instances) == spec['instances'][flow['name']]
                    for instance in instances:
                        mask = sel & (values['instance'] == instance)
                        assert mask.sum() == 10
                        per_gt.append(dict(seed=seed, flow=flow['name'], instance=int(instance),
                                           samples=10, correct=int(np.sum(p[mask] >= .5)), missed=int(np.sum(p[mask] < .5))))
                results.append(dict(seed=seed, scope='complete_test', **score))
                results.append(dict(seed=seed, scope='original_test', **row['original_test']))
        per_seed.append(row)
    reference_path = source/'final/c156/seed96721/test_predictions.npz'
    assert sha(reference_path) == spec['c156_reference_prediction_sha256']
    with np.load(reference_path) as z:
        reference = metrics(z['labels'], z['probability'])
    aggregate = {}
    for scope, metric in (('test', 'f1'), ('original_test', 'f1'), ('added_head', 'recall')):
        values = [r[scope][metric] for r in per_seed]
        aggregate[scope] = dict(metric=metric, mean=float(np.mean(values)), std=float(np.std(values, ddof=1)), seeds=values)
    report = dict(complete=True, config_sha256=sha(config), scientific_commit=commit, source_files_verified=18,
                  independent_prediction_count=6, per_seed=per_seed, aggregate=aggregate,
                  c156_single_seed_reference=dict(seed=96721, test=reference,
                      p35_minus_c156_same_seed=per_seed[0]['test']['f1']-reference['f1']))
    (root/'independent_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    for name, rows in (('per_scope.csv', results), ('per_GT.csv', per_gt)):
        with (root/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    text = '# Verify_Task4C_P35GTHead_1.1\n\nOriginal p35/h0, 76,738 parameters; fixed expanded data.\n\n'
    text += '| Seed | Validation F1 | Full test F1 | Original test F1 | Added head recall | Training minutes |\n|---|---|---|---|---|---|\n'
    for r in per_seed:
        text += f"| {r['seed']} | {r['validation']['f1']:.6f} | {r['test']['f1']:.6f} | {r['original_test']['f1']:.6f} | {r['added_head']['recall']:.6f} | {r['training_seconds']/60:.3f} |\n"
    for scope, item in aggregate.items(): text += f"\n{scope}: {item['mean']:.6f} +/- {item['std']:.6f}.\n"
    text += '\nThe c156 comparison is single-seed (96721), not a paired three-seed method comparison.\n'
    (root/'report.md').write_text(text)
    print(json.dumps(report['aggregate']), flush=True)


if __name__ == '__main__':
    import sys
    config = sys.argv[1]
    audit(json.loads(Path(config).read_text()), config)
