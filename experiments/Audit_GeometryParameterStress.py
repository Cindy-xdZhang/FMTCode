"""Independently recompute diagnostic metrics; no training/encoder imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import (f1_score, average_precision_score, jaccard_score,
                             precision_score, recall_score, balanced_accuracy_score)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as handle: return list(csv.DictReader(handle))


def audit(spec, config):
    root = Path(spec['output_root']); audited = []; count = 0
    config_sha = sha(config)
    for task in spec['tasks']:
        for dataset in spec['datasets']:
            shared_labels = shared_identities = None
            for seed in spec['seeds'][task]:
                target = root/'shards'/task/dataset/f'seed{seed}'
                done = json.loads((target/'complete.json').read_text())
                frozen = json.loads((target/'frozen_before_confirmation.json').read_text())
                if done['status'] != 'COMPLETE' or done['config_sha256'] != config_sha:
                    raise ValueError('incomplete/mismatched shard')
                if frozen['config_sha256'] != config_sha or frozen['confirmation_metrics_computed']:
                    raise ValueError('invalid freeze barrier record')
                if sha(target/'predictions.npz') != done['predictions_sha256'] or sha(target/'per_run.csv') != done['per_run_sha256']:
                    raise ValueError('modified evidence')
                if list((target/'temporary_training').rglob('*.pt')):
                    raise ValueError('temporary checkpoints remain')
                with np.load(target/'predictions.npz', allow_pickle=False) as z:
                    labels = z['labels']; scores = z['scores']; thresholds = z['thresholds']
                    names = z['case_ids'].tolist(); identities = str(z['identities_json'])
                if shared_labels is None: shared_labels, shared_identities = labels.copy(), identities
                if not np.array_equal(labels, shared_labels) or identities != shared_identities:
                    raise ValueError('population changed across seeds')
                rows = read_csv(target/'per_run.csv')
                keyed = {r['case']+'/'+r['arm']: r for r in rows}
                expected = {v['id']+'/'+a for v in spec['geometry_variants'] for a in ('direct', 'learned')}
                expected |= {f'threshold_offset_{v:g}/'+a for v in spec['decision_offsets_in_train_score_std'] for a in ('direct', 'learned')}
                expected |= {v['id']+'/learned' for v in spec['training_variants'][task] if v['id'] != 'reference'}
                if set(names) != expected or set(keyed) != expected or len(names) != len(expected) or len(rows) != len(expected):
                    raise ValueError('missing/duplicate/unexpected case')
                if scores.shape != (len(expected), len(labels)) or not np.isfinite(scores).all():
                    raise ValueError('invalid prediction matrix')
                for index, name in enumerate(names):
                    row = keyed[name]; arm = row['arm']; case = row['case']
                    meta = frozen['direct'] if arm == 'direct' else frozen['training']['reference']
                    expected_threshold = meta['threshold']
                    if row['stage'] == 'decision_threshold_swap':
                        expected_threshold += float(row['threshold_offset'])*meta['score_std']
                        original = scores[names.index('reference/'+arm)]
                        np.testing.assert_array_equal(original, scores[index])
                    elif row['stage'] == 'training_parameter_swap':
                        expected_threshold = frozen['training'][case]['threshold']
                    if thresholds[index] != expected_threshold or float(row['threshold']) != expected_threshold:
                        raise ValueError('threshold was not the predeclared value')
                    decision = scores[index] >= thresholds[index]; y = labels.astype(bool)
                    values = {'f1': f1_score(y, decision, zero_division=0),
                        'average_precision': average_precision_score(y, scores[index]) if y.any() else 0.,
                        'iou': jaccard_score(y, decision, zero_division=0),
                        'precision': precision_score(y, decision, zero_division=0),
                        'recall': recall_score(y, decision, zero_division=0),
                        'balanced_accuracy': balanced_accuracy_score(y, decision),
                        'predicted_positive_fraction': float(decision.mean()),
                        'tp': int(np.sum(y & decision)), 'fp': int(np.sum(~y & decision)),
                        'tn': int(np.sum(~y & ~decision)), 'fn': int(np.sum(y & ~decision))}
                    for metric, value in values.items():
                        if not np.isclose(float(row[metric]), value, rtol=0, atol=1e-8):
                            raise ValueError(f'incorrect metric {target}/{name}/{metric}')
                    for metric in ('f1', 'average_precision', 'iou', 'balanced_accuracy'):
                        drop = float(keyed['reference/'+arm][metric])-values[metric]
                        if not np.isclose(drop, float(row[metric+'_drop_from_reference']), rtol=0, atol=1e-8):
                            raise ValueError('incorrect paired degradation')
                    count += 1
                audited.extend(rows)
    table = read_csv(root/'parameter_stress_table.csv')
    keys = {(r['task'], r['case'], r['arm'], r['stage']) for r in audited}
    if len(table) != len(keys) or {(r['task'], r['case'], r['arm'], r['stage']) for r in table} != keys:
        raise ValueError('incomplete summary')
    fields = [k for k in table[0] if k not in {'task', 'case', 'arm', 'stage'} and not k.endswith('_dataset_std')]
    for row in table:
        subset = [r for r in audited if all(r[k] == row[k] for k in ('task', 'case', 'arm', 'stage'))]
        for metric in fields:
            values = [np.mean([float(r[metric]) for r in subset if r['dataset'] == d]) for d in spec['datasets']]
            if not np.isclose(float(row[metric]), np.mean(values), rtol=0, atol=1e-8): raise ValueError('incorrect macro mean')
            if not np.isclose(float(row[metric+'_dataset_std']), np.std(values, ddof=1), rtol=0, atol=1e-8):
                raise ValueError('incorrect dataset spread')
    result = {'status': 'PASS', 'purpose': spec['purpose'], 'checked_rows': count,
              'config_sha256': config_sha, 'table_sha256': sha(root/'parameter_stress_table.csv')}
    (root/'independent_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/Verify_Task123_GeometryParameterStress_1.1.yaml')
    args = parser.parse_args(); audit(yaml.safe_load(Path(args.config).read_text()), args.config)
