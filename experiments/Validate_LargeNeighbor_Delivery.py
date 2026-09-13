"""Validate downloaded scalar results against hashes and per-seed records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def validate(root, config):
    spec = json.loads(config.read_text(encoding='utf-8'))
    verified = []
    for line in (root / 'METRIC_DELIVERY.sha256').read_text().splitlines():
        expected, name = line.split(None, 1)
        name = name.lstrip('* ')
        target = (root / name).resolve()
        assert target.parent == root.resolve(), name
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        assert actual == expected, name
        assert target.suffix in {'.csv', '.json', '.md', '.psv', '.jsonl'}
        verified.append(name)
    audit = json.loads((root / 'independent_audit.json').read_text())
    assert audit['status'] == 'PASS' and audit['shards'] == 150
    assert audit['metric_rows'] == 950 and audit['scale_rows'] == 0
    cleanup = json.loads((root / 'checkpoint_cleanup_summary.json').read_text())
    assert cleanup['remaining'] == 0 and cleanup['deleted_count'] == 350
    rows = read_rows(root / 'per_run.csv')
    assert len(rows) == 950
    lookup = {(r['task'], r['dataset'], int(r['seed']), r['arm']): r for r in rows}
    assert len(lookup) == len(rows)
    for task in spec['tasks']:
        part = spec[task.lower()]
        for dataset in spec['datasets']:
            for seed in part['seeds']:
                for arm in part['arms']:
                    assert (task, dataset, seed, arm) in lookup
    comparisons = 0
    for kind in ('dataset_metrics', 'task_macro'):
        for row in read_rows(root / (kind + '.csv')):
            task, arm = row['task'], row['arm']
            for field, expected in row.items():
                if not field.endswith(('_mean', '_std')) or expected == '':
                    continue
                metric, summary = field.rsplit('_', 1)
                values = []
                for seed in spec[task.lower()]['seeds']:
                    datasets = [row['dataset']] if kind == 'dataset_metrics' else spec['datasets']
                    values.append(statistics.mean(float(lookup[task, d, seed, arm][metric]) for d in datasets))
                actual = statistics.mean(values) if summary == 'mean' else statistics.stdev(values)
                assert abs(actual - float(expected)) < 1e-10, (kind, row, field, actual)
                comparisons += 1
    for row in read_rows(root / 'paired_comparisons.csv'):
        task, dataset = row['task'], row['dataset']
        for field, expected in row.items():
            if not field.startswith('delta_') or expected == '':
                continue
            metric, summary = field[6:].rsplit('_', 1)
            values = [float(lookup[task, dataset, seed, row['method']][metric]) -
                      float(lookup[task, dataset, seed, row['comparator']][metric])
                      for seed in spec[task.lower()]['seeds']]
            actual = statistics.mean(values) if summary == 'mean' else statistics.stdev(values)
            assert abs(actual - float(expected)) < 1e-10, (row, field, actual)
            comparisons += 1
    sampling = read_rows(root / 'sampling_audit.csv')
    cache = json.loads((root / 'cache_integrity_audit.json').read_text())
    assert cache['status'] == 'PASS' and cache['files'] == len(sampling)
    assert sum(int(r['common_valid_count']) for r in sampling) == cache['common_samples']
    for row in sampling:
        if row['dataset'] in spec['cylinder_datasets']:
            assert float(row['initial_time']) >= 7.5
        assert int(row['old_valid_count']) - int(row['common_valid_count']) == int(row['newly_excluded'])
    result = {'status': 'PASS', 'verified_files': verified, 'metric_rows': len(rows),
              'aggregate_statistics_recomputed': comparisons, 'sampling_files': len(sampling),
              'scope': 'Local scalar/hash audit; prediction recomputation was performed on Ibex.',
              'checkpoint_files_downloaded': 0}
    (root / 'local_delivery_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/Verify_LargeNeighbor_1.1'))
    parser.add_argument('--config', type=Path, default=Path('config/Verify_LargeNeighbor_1.1.json'))
    args = parser.parse_args()
    validate(args.root, args.config)
