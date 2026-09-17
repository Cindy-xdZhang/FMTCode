"""Independently verify downloaded c156 predictions and viewer export rows."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4194304), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def arrays(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def score(y, probability):
    y = np.asarray(y).astype(bool)
    p = np.asarray(probability) >= .5
    tp, fp, fn, tn = [int(np.sum(m)) for m in (y & p, ~y & p, y & ~p, ~y & ~p)]
    return dict(f1=2*tp/max(2*tp+fp+fn, 1), accuracy=(tp+tn)/len(y),
                precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
                average_precision=float(average_precision_score(y, probability)),
                roc_auc=float(roc_auc_score(y, probability)), true_positive=tp,
                false_positive=fp, false_negative=fn, true_negative=tn, samples=len(y))


def audit(root, config, catalog):
    root = Path(root)
    spec = read(config)
    run = root/'final/c156/seed96721'
    result = read(run/'result.json')
    package = root/'viewer_package'
    manifest = read(package/'manifest.json')
    completion = read(root/'completion.json')
    data_check = read(root/'data_audit.json')
    gpu_check = read(root/'gpu_check.json')
    before = read(root/'selection.lock.json')
    selection = read(run/'selection.lock.json')
    assert all(r['complete'] for r in (result, completion, data_check, gpu_check, before))
    assert result['candidate'] == spec['candidate'] == before['selected'] == selection['candidate']
    assert result['seed'] == 96721 and result['parameters'] == 992386
    assert spec['candidate']['augmentation'] == 'none' and spec['final']['seeds'] == [96721]
    identity = result['identity']
    assert identity['config_sha256'] == sha(config)
    for file, digest in identity['sources'].items():
        blob = subprocess.check_output(['git', 'cat-file', 'blob', identity['git_commit']+':'+file])
        assert hashlib.sha256(blob).hexdigest() == digest, file
    for r in (completion, data_check, gpu_check, before, selection, manifest):
        assert r['identity'] == identity
    assert completion['package_manifest_sha256'] == sha(package/'manifest.json')
    assert data_check['counts'] == spec['expected_counts']
    assert data_check['original_prefix_unchanged'] and data_check['old_validation_unchanged']
    history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    assert history == result['history'] and len(history) == result['epochs']
    assert [r['epoch'] for r in history] == list(range(1, len(history)+1))
    assert all(r['samples'] == 196960 and np.isfinite(r['training_loss']) for r in history)
    best = max(history, key=lambda r: (r['validation_f1'], r['validation_average_precision']))
    assert best['epoch'] == result['selected_epoch'] == selection['selected_epoch']
    assert len(history) == 500 or len(history)-best['epoch'] == 50
    assert selection['threshold'] == .5 and not selection['test_loaded']
    norm = result['normalization']
    assert norm['train_only'] and np.isfinite(norm['mean']).all()
    assert np.isfinite(norm['std']).all() and np.all(np.asarray(norm['std']) > 0)
    reports, heads = {}, {}
    original_test_labels, original_test_probability = [], []
    gt = read(catalog)['flows']
    for role in ('train', 'validation', 'test'):
        file = run/(role+'_predictions.npz')
        assert sha(file) == result['predictions'][role]
        p = arrays(file)
        assert len(p['labels']) == spec['expected_counts'][role]
        assert p['threshold'] == .5
        assert np.isfinite(p['probability']).all() and np.all((p['probability'] >= 0) & (p['probability'] <= 1))
        metric = score(p['labels'], p['probability'])
        expected = (result['validation'] if role == 'validation' else result['test']['combined']) if role != 'train' else manifest['models'][0]['metrics']['train']['pooled']
        for key, value in expected.items():
            assert abs(metric[key]-value) < 1e-12, (role, key)
        reports[role] = dict(combined=metric, per_flow={})
        for fi, flow in enumerate(spec['flows']):
            name = flow['name']
            folder = root/'physical'/name
            m = arrays(folder/role/'metadata.npz')
            preparation = read(folder/'preparation.json')
            assert sha(folder/role/'metadata.npz') == preparation['splits'][role]['files']['metadata.npz']
            sel = p['flow_index'] == fi
            rows = p['row_in_split'][sel]
            assert np.array_equal(rows, np.arange(len(m['labels'])))
            for key in ('labels', 'instance'):
                assert np.array_equal(p[key][sel], m[key])
            probability = p['probability'][sel]
            reports[role]['per_flow'][name] = score(m['labels'], probability)
            if role == 'validation':
                continue
            entry = manifest['splits'][name+'/'+role]
            exported = package/entry['file']
            assert sha(exported) == entry['sha256'] and entry['population'] == len(rows)
            z = arrays(exported)
            ids = z['row_ids']
            assert len(ids) == entry['selected'] and len(np.unique(ids)) == len(ids)
            assert np.array_equal(z['p_c156'], probability[ids])
            for key in ('labels', 'counts', 'center', 'instance', 'scale_id', 'radius', 'neighbor_distance'):
                assert np.array_equal(z[key], m[key][ids])
            assert np.isfinite(z['geometry']).all()
            valid = np.arange(27)[None] < z['counts'][:, None]
            assert np.all(z['geometry'][~valid] == 0)
            added = z['mandatory_head'].astype(bool)
            instance_ids, counts = np.unique(z['instance'][added], return_counts=True)
            expected_ids = sorted(row['instance'] for row in gt[name]['instances'])
            assert instance_ids.tolist() == expected_ids
            assert np.all(counts == (30 if role == 'train' else 10))
            assert np.all(z['labels'][added] == 1)
            if role == 'train':
                assert added.all() and np.array_equal(ids, np.arange(len(rows)-len(ids), len(rows)))
            else:
                assert np.array_equal(ids, rows)
                old_n = len(rows)-int(added.sum())
                assert np.array_equal(added, rows >= old_n)
                original_test_labels.append(m['labels'][:old_n])
                original_test_probability.append(probability[:old_n])
                heads[name] = dict(old_test=score(m['labels'][:old_n], probability[:old_n]),
                    added=int(added.sum()), correct=int(np.sum(probability[added] >= .5)),
                    missed=int(np.sum(probability[added] < .5)),
                    per_instance={str(i): dict(total=10, correct=int(np.sum(probability[added & (m['instance'] == i)] >= .5))) for i in instance_ids})
                expected_head = manifest['head_coverage'][name]
                assert abs(heads[name]['correct']/heads[name]['added']-expected_head['added_head_recall']) < 1e-12
                for i, h in heads[name]['per_instance'].items():
                    assert h['correct'] == expected_head['added_head_counts'][i]['correctly_predicted']
    runtime = [json.loads(line) for line in (root/'runtime.jsonl').read_text().splitlines()]
    processes = {}
    for event in runtime:
        assert event['identity'] == identity
        processes.setdefault((event['job_id'], event['array_index'], event['phase']), []).append(event)
    assert len(processes) == 6 and len(runtime) == 12
    for events in processes.values():
        assert [r['state'] for r in events] == ['STARTED', 'ENDED']
        assert events[-1]['exit_code'] == 0
    output = dict(complete=True, identity=identity, seed=96721, parameters=992386,
        audit_implementation=dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), source_sha256=sha(__file__)),
        selected_epoch=best['epoch'], epochs=len(history), training_seconds=result['training_seconds'],
        metrics=reports, head_coverage=heads, runtime_processes=len(processes), runtime_events=len(runtime),
        original_test_combined=score(np.concatenate(original_test_labels), np.concatenate(original_test_probability)),
        source_result_sha256=sha(run/'result.json'), single_seed=True)
    path = root/'independent_results_audit.json'
    path.write_text(json.dumps(output, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(complete=True, audit=str(path), test_f1=reports['test']['combined']['f1'], heads=heads)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='outputs/mainExp_Task4C_GTHeadCoverage_1.1')
    p.add_argument('--config', default='outputs/mainExp_Task4C_GTHeadCoverage_1.1/scientific_config.json')
    p.add_argument('--catalog', default='outputs/Verify_Task4C_GTHeadCoverage_1.1/catalog.json')
    a = p.parse_args()
    audit(a.root, a.config, a.catalog)
