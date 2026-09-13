"""Independently recompute metrics and audit every registered transfer run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/Verify_AIVDTransfer_1.1.json')
    args = parser.parse_args()
    spec = read_json(args.config)
    root = Path(spec['output_root'])
    checked, all_rows, devices, parameters, diagnostics = [], [], [], [], []
    expected_shards = sum(len(spec['datasets'])*len(spec[t.lower()]['seeds']) for t in spec['tasks'])
    expected_rows = sum(len(spec['datasets'])*len(spec[t.lower()]['seeds'])*len(spec[t.lower()]['arms']) for t in spec['tasks'])
    preflight = read_json(root/'preflight.json')
    assert preflight['status'] == 'PASS' and not preflight['confirmation_opened']
    assert preflight['config_sha256'] == digest(args.config)
    for task in spec['tasks']:
        part = spec[task.lower()]
        for dataset in spec['datasets']:
            reference_hash = None
            reference_inputs = None
            for seed in part['seeds']:
                folder = root/'shards'/task/dataset/f'seed{seed}'
                done = read_json(folder/'complete.json')
                assert done['status'] == 'COMPLETE'
                assert done['config_sha256'] == digest(args.config)
                assert done['source_manifest_sha256'] == preflight['source_manifest_sha256']
                assert done['prediction_sha256'] == digest(folder/'predictions.npz')
                assert done['per_run_sha256'] == digest(folder/'per_run.csv')
                assert done['checkpoints_written'] == 0
                inputs = read_json(folder/'input_audit.json')
                test = read_json(folder/'confirmation_input_audit.json')
                roles = {'train': inputs['training'], 'validation': inputs['validation'], 'confirmation': test}
                groups = [{r['path'] for r in records} for records in roles.values()]
                assert not any(groups[i]&groups[j] for i, j in ((0, 1), (0, 2), (1, 2)))
                for role, records in roles.items():
                    ordinals = spec.get('dataset_splits', {}).get(dataset, {}).get(task, {}).get(role, part[role])
                    assert [r['ordinal'] for r in records] == ordinals
                    if dataset in spec['cylinder_datasets']:
                        assert all(float(r['metadata']['source_time']) >= spec['minimum_cylinder_time'] for r in records)
                signature = [[(r['path'], r['sha256'], r['identity'], r['raw_sha256'], r['times_sha256'])
                              for r in records] for records in roles.values()]
                if reference_inputs is None:
                    reference_inputs = signature
                assert signature == reference_inputs
                frozen = read_json(folder/'frozen_models.json')
                assert sorted(frozen) == sorted(part['arms'])
                for arm in part['arms']:
                    state = frozen[arm]
                    assert state['input_dimension'] == spec['feature_dimensions'][arm]
                    assert state['feature'] == spec['features'][arm]
                    if task == 'Task1':
                        assert state['pca_dimension'] == (None if arm == 'aivd' else 8)
                    else:
                        assert state['losses']['completed_optimizer_steps'] == part['architecture']['optimizer_steps'] == 7000
                        parameters.append({'task': task, 'dataset': dataset, 'seed': seed, 'arm': arm,
                                           'input_dimension': state['input_dimension'],
                                           **state['losses']})
                with (folder/'per_run.csv').open() as f:
                    rows = list(csv.DictReader(f))
                assert sorted(r['arm'] for r in rows) == sorted(part['arms'])
                with np.load(folder/'predictions.npz', allow_pickle=False) as z:
                    y = z['labels'].astype(bool)
                    assert len(y) == sum(r['valid'] for r in test)
                    h = hashlib.sha256(z['labels'].tobytes()).hexdigest()
                    if reference_hash is None:
                        reference_hash = h
                    assert reference_hash == h
                    for row in rows:
                        assert row['task'] == task and row['dataset'] == dataset and int(row['seed']) == seed
                        p = z[f"prediction_{row['arm']}"].astype(bool)
                        assert p.shape == y.shape
                        tp, fp = int(np.sum(y&p)), int(np.sum(~y&p))
                        fn, tn = int(np.sum(y&~p)), int(np.sum(~y&~p))
                        metrics = {
                            'f1': 2*tp/max(2*tp+fp+fn, 1), 'iou': tp/max(tp+fp+fn, 1),
                            'precision': tp/max(tp+fp, 1), 'recall': tp/max(tp+fn, 1),
                            'balanced_accuracy': .5*(tp/max(tp+fn, 1)+tn/max(tn+fp, 1)),
                            'ari': float(adjusted_rand_score(y, p)),
                            'nmi': float(normalized_mutual_info_score(y, p)),
                            'positive_fraction': float(y.mean()), 'predicted_positive_fraction': float(p.mean()),
                        }
                        for key, value in metrics.items():
                            assert abs(value-float(row[key])) < 1e-12, (folder, row['arm'], key)
                        assert int(row['sample_count']) == len(y)
                        all_rows.append({**row, 'true_positive': tp, 'false_positive': fp,
                                         'false_negative': fn, 'true_negative': tn})
                assert not any(p.suffix.lower() in ('.pt', '.pth', '.ckpt') for p in folder.rglob('*') if p.is_file())
                devices.append(done)
                diagnostics.extend({'task': task, 'dataset': dataset, 'seed': seed, **r}
                                   for r in inputs['feature_diagnostics'])
                checked.append({'task': task, 'dataset': dataset, 'seed': seed, 'rows': len(rows)})
    assert len(checked) == expected_shards == 100
    assert len(all_rows) == expected_rows == 350
    write_csv(root/'per_run.csv', all_rows)
    write_csv(root/'execution_records.csv', devices)
    write_csv(root/'training_diagnostics.csv', parameters)
    write_csv(root/'feature_diagnostics.csv', diagnostics)
    output = {
        'status': 'PASS', 'experiment': spec['experiment'], 'config_sha256': digest(args.config),
        'audit_source_sha256': digest(__file__),
        'shards': len(checked), 'metric_rows': len(all_rows),
        'per_run_sha256': digest(root/'per_run.csv'),
        'checks': ['prediction_and_metric_hashes', 'config_and_source_manifest', 'identical_inputs_across_seeds',
                   'independent_confusion_matrices_ARI_NMI', 'identical_labels_across_arms_and_seeds',
                   'disjoint_split_paths_and_registered_ordinals', 'all_cylinder_roles_t>=7.5',
                   'exact_feature_dimensions_and_Task1_PCA_policy', 'Task2_exact_7000_updates',
                   'all_baselines_retrained_in_same_shard', 'no_model_checkpoints'],
        'all_shards': checked,
    }
    (root/'independent_audit.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'all_shards'}, indent=2))


if __name__ == '__main__':
    main()
