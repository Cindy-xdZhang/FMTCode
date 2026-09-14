"""Independent source-label, capacity, paired-summary and retention checks.

Read only saved predictions, frozen model metadata and original source arrays;
never import a training runner or open a checkpoint.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    config = read_json(root/'run_config.json')
    assert read_json(root/'independent_audit.json')['status'] == 'PASS'
    checked_sources, capacities, checks = {}, [], []
    for task in config['tasks']:
        for dataset in config['datasets']:
            for seed in config[task.lower()]['seeds']:
                directory = root/'shards'/task/dataset/f'seed{seed}'
                completed = read_json(directory/'complete.json')
                assert completed['config_sha256'] == digest(root/'run_config.json')
                train = read_json(directory/'input_audit.json')
                evaluation = read_json(directory/'confirmation_input_audit.json')
                source_sets = [{r['path'] for r in rows} for rows in
                               [train['training'],train['validation'],evaluation]]
                assert not any(source_sets[i]&source_sets[j] for i,j in [(0,1),(0,2),(1,2)])
                for record in train['training']+train['validation']+evaluation:
                    name = record['path']
                    if name not in checked_sources:
                        assert digest(name) == record['sha256']
                        with np.load(name, allow_pickle=False) as source:
                            checked_sources[name] = (source['reference'].astype(np.float32),
                                source['scale_id'] if 'scale_id' in source else None)
                y = np.concatenate([checked_sources[r['path']][0] for r in evaluation])
                with np.load(directory/'predictions.npz', allow_pickle=False) as predicted:
                    np.testing.assert_array_equal(predicted['labels'], y)
                    if task == 'Task5':
                        scale = np.concatenate([checked_sources[r['path']][1] for r in evaluation])
                        np.testing.assert_array_equal(predicted['scale_id'], scale)
                assert completed['models_frozen_before_test_time'] <= (directory/'confirmation_input_audit.json').stat().st_mtime + 1
                frozen = read_json(directory/'frozen_models.json')
                if task == 'Task2':
                    assert all(v['losses']['completed_optimizer_steps']==7000 for v in frozen.values())
                    assert {v['input_dimension'] for v in frozen.values()} == {700}
                    assert len({v['losses']['parameter_count'] for v in frozen.values()}) == 1
                    for arm, values in frozen.items():
                        capacities.append({'task':task,'dataset':dataset,'seed':seed,'arm':arm,
                            'parameter_count':values['losses']['parameter_count'],'trainable_parameter_count':values['losses']['parameter_count']})
                if task in ['Task3','Task5']:
                    auxiliaries = {k:v for k,v in frozen.items() if k not in ['raw','raw_wide','fixed_task3_raw']}
                    assert {v['input_dimension'] for v in auxiliaries.values()} == {433}
                    assert len({v['parameter_count'] for v in auxiliaries.values()}) == 1
                    assert len({v['trainable_parameter_count'] for v in auxiliaries.values()}) == 1
                    assert next(iter(auxiliaries.values()))['parameter_count'] < frozen['raw_wide']['parameter_count']
                    for arm, values in frozen.items():
                        capacities.append({'task':task,'dataset':dataset,'seed':seed,'arm':arm,
                            'parameter_count':values['parameter_count'],
                            'trainable_parameter_count':values.get('trainable_parameter_count',values['parameter_count'])})
                checks.append({'task':task,'dataset':dataset,'seed':seed,'source_labels_equal':True,'samples':len(y)})
    rows = list(csv.DictReader((root/'per_run_metrics.csv').open()))
    macro, paired = [], []
    for task in config['tasks']:
        part = config[task.lower()]
        for metric in ['f1','average_precision']:
            if task in ['Task1','Task2'] and metric=='average_precision':
                continue
            for arm in part['arms']:
                values = [np.mean([float(r[metric]) for r in rows
                          if r['task']==task and r['arm']==arm and int(r['seed'])==seed]) for seed in part['seeds']]
                assert np.isfinite(values).all()
                macro.append({'task':task,'arm':arm,'metric':metric,'mean':float(np.mean(values)),
                              'seed_macro_std':float(np.std(values,ddof=1)),'seed_count':len(values)})
            for method, parent in [('fmt_v5','old_fmt'),('fmt_objective_ntod_v2','ntdo_v2')]:
                delta = []
                for seed in part['seeds']:
                    by_arm = {arm:np.mean([float(r[metric]) for r in rows if r['task']==task and r['arm']==arm and int(r['seed'])==seed]) for arm in [method,parent]}
                    delta.append(by_arm[method]-by_arm[parent])
                positive = 0
                for dataset in config['datasets']:
                    by_arm = {arm:np.mean([float(r[metric]) for r in rows if r['task']==task and r['arm']==arm and r['dataset']==dataset]) for arm in [method,parent]}
                    positive += by_arm[method]>by_arm[parent]
                paired.append({'task':task,'method':method,'parent':parent,'metric':metric,
                               'mean_gain':float(np.mean(delta)),'seed_macro_gain_std':float(np.std(delta,ddof=1)),
                               'positive_datasets':int(positive),'dataset_count':len(config['datasets'])})
    remaining = [str(p) for ext in ['*.pt','*.pth','*.ckpt'] for p in root.rglob(ext)]
    assert not remaining, remaining
    write_csv(root/'macro_summary.csv', macro)
    write_csv(root/'paired_macro_summary.csv', paired)
    write_csv(root/'network_parameters.csv', capacities)
    result = {'status':'PASS','shards':len(checks),'original_sources_checked':len(checked_sources),
              'source_labels_and_task5_scales_match':True,'capacity_checks_pass':True,
              'remaining_checkpoints':remaining,'checks':checks,'auditor_sha256':digest(__file__),
              'macro_summary_sha256':digest(root/'macro_summary.csv'),
              'paired_macro_summary_sha256':digest(root/'paired_macro_summary.csv')}
    (root/'source_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


if __name__ == '__main__':
    main()
