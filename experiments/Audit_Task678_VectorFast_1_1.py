"""Same directional-token validation rules with independent Euclidean pair distances."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from FMT_Utils.FlowMapData_3D import sha256, write_json
from FMT_Utils.HanFlowMapData_3D import ROLES, TEST_ROLES, support_features
from FMT_Utils.FlowMapAuditMetrics_3D import measured_trajectories
from experiments.Build_Task678_FlowMap_1_1 import provenance
from experiments.Run_Task678_VectorFMT_1_1 import DEFAULT_CONFIG
from experiments.Run_Task678_VectorFMT_1_1 import load_role, matrix, derangement_by_window
from experiments.Run_Task678_FlowMap_1_1 import truth_radius


def compare(expected, actual):
    for k, v in expected.items():
        if isinstance(v, str):
            assert actual[k] == v, k
        else:
            np.testing.assert_allclose(v, actual[k], rtol=1e-9, atol=1e-9, err_msg=k)



def audit(spec, config):
    spec['_config_sha256'] = sha256(config)
    root = Path(spec['output_root']); rows = []
    assert json.loads((Path(spec['base_output_root']) / 'data_audit.json').read_text())['status'] == 'PASS'
    for dataset in spec['datasets']:
        arms = [(arm, seed) for seed in spec['seeds'] for arm in spec['neural_arms']] + [('vector_affine', 0)]
        for arm, seed in arms:
            directory = root / 'runs' / dataset / arm / f'seed{seed}'
            done = json.loads((directory / 'completed.json').read_text())
            assert done['status'] == 'PASS' and done['config_sha256'] == sha256(config)
            assert done['frozen_encoder_sha256'] == spec['frozen_encoder_sha256']
            assert done['vector_encoder_sha256'] == spec['vector_encoder_sha256']
            assert done['base_config_sha256'] == spec['base_config_sha256']
            assert done['metric_records'] == (30 if arm in ('vector_fmt6',) else 21)
            if arm != 'vector_affine':
                for task in ('Task6', 'Task7'):
                    train = json.loads((directory / f'{task}_training.json').read_text())
                    s = spec['decoder']
                    expected = math.ceil(s['equivalent_epochs'] * train['training_examples'] * s_train_q(spec) * 31 / (s['bundle_batch']*s['queries_per_bundle']))
                    assert train['optimizer_steps'] == s.get('smoke_steps', expected)
                    assert train['supervised_time_indices'] == list(range(2, 63, 2))
                    assert train['withheld_time_indices'] == list(range(1, 63, 2))
                    assert train['input_dimension'] == 231
                    assert train['projection'] == 'none'
        for role in ROLES:
            data, sources = load_role(spec, dataset, role)
            n = len(data['origin0'])
            for arm, seed in arms:
                directory = root / 'runs' / dataset / arm / f'seed{seed}'
                variants = ['normal'] + (['shuffled_tokens'] if role in TEST_ROLES and arm in ('vector_fmt6',) else [])
                for variant in variants:
                    for task in ('Task6', 'Task7', 'Task8'):
                        record = json.loads((directory / f'{task}_{role}_{variant}.json').read_text())
                        assert record['source_records'] == sources and record['config_sha256'] == sha256(config)
                        path = directory / record['prediction_file']
                        assert sha256(path) == record['prediction_sha256']
                        with np.load(path, allow_pickle=False) as archive:
                            prediction = archive['prediction']
                            truth, radius = truth_radius(data, task)
                            measured = measured_trajectories(prediction, truth, radius)
                            compare(measured, record['metrics'])
                            for ordinal in np.unique(data['window_ordinal']):
                                ix = np.flatnonzero(data['window_ordinal'] == ordinal)
                                if task == 'Task6':
                                    ix = np.concatenate([ix, n + ix])
                                compare(measured_trajectories(prediction[ix], truth[ix], radius[ix]), record['per_window'][str(ordinal)])
                            if task == 'Task8':
                                assert record['second_query_uses'] == 'predicted_first_stage_endpoint'
                                arrival = (prediction[:, :, 62] - data['origin1'][:, None]) / data['radius1'][:, None, None]
                                np.testing.assert_allclose((np.abs(arrival).sum(-1) > 1).mean(), record['predicted_arrival_outside_fraction'])
                                if arm != 'vector_affine' and variant == 'normal':
                                    compare(measured_trajectories(archive['true_arrival_diagnostic'], data['target_long'][:, :, 62:], data['radius1']), record['true_arrival_diagnostic_metrics'])
                        if variant == 'shuffled_tokens':
                            np.testing.assert_array_equal(record['token_permutation'], derangement_by_window(data, seed))
                        row = dict(dataset=dataset, family=spec['families'][dataset], arm=arm, seed=seed, task=task,
                            role=role, variant=variant, **{k:v for k,v in measured.items() if isinstance(v,(float,int))},
                            parameters=record['parameter_count'], steps=record['optimizer_steps'], token_bytes=record['token_bytes'],
                            initial_train_probe=record.get('initial_train_probe_nrmse'), final_train_probe=record.get('final_train_probe_nrmse'),
                            predicted_arrival_outside_fraction=record.get('predicted_arrival_outside_fraction'),
                            true_arrival_diagnostic_nrmse=record.get('true_arrival_diagnostic_metrics',{}).get('position_nrmse'),
                            prediction_sha256=record['prediction_sha256'])
                        rows.append(row)
            print('METRIC REPLAY PASS', dataset, role, flush=True)
    assert not any(root.rglob('*.pt')) and not any(root.rglob('*.pth')) and not any(root.rglob('*.ckpt'))
    out = root / 'metrics.csv'
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row['arm'], row['task'], row['role'], row['variant']), []).append(row)
    summaries = []
    for (arm, task, role, variant), values in groups.items():
        per_dataset = {d:float(np.mean([r['position_nrmse'] for r in values if r['dataset']==d])) for d in spec['datasets']}
        per_family = {f:float(np.mean([per_dataset[d] for d in spec['datasets'] if spec['families'][d]==f])) for f in set(spec['families'].values())}
        per_seed = {str(seed):float(np.mean([r['position_nrmse'] for r in values if r['seed']==seed])) for seed in sorted({r['seed'] for r in values})}
        summaries.append(dict(arm=arm,task=task,role=role,variant=variant,per_dataset=per_dataset,per_family=per_family,
            dataset_macro_position_nrmse=float(np.mean(list(per_dataset.values()))),
            family_macro_position_nrmse=float(np.mean(list(per_family.values()))), seed_macro=per_seed,
            seed_macro_std=float(np.std(list(per_seed.values()),ddof=1)) if len(per_seed)>1 else None))
    write_json(root / 'summary.json', {**provenance(config),'status':'PASS','summary':summaries})
    write_json(root / 'independent_audit.json', {**provenance(config),'status':'PASS','metric_records':len(rows),
        'checkpoint_files':0,'metrics_sha256':sha256(out),'summary_sha256':sha256(root/'summary.json')})
    print('VECTOR FMT AUDIT PASS',len(rows),'records',flush=True)


def s_train_q(spec):
    return spec['sampling']['train_queries']


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default=DEFAULT_CONFIG)
    a=p.parse_args(); spec=json.loads(Path(a.config).read_text())
    audit(spec,a.config)


if __name__ == '__main__':
    main()
