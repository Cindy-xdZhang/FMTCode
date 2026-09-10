"""Dense-query paired experiments; no test choice, no saved model weights."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256, write_json, affine_query
from FMT_Utils.FlowMapModels_3D import training_arrays
from FMT_Utils.FlowMapFit_3D import predict_direct, compose_direct
from FMT_Utils.HanFlowMapFit_3D import fit_dense
from FMT_Utils.HanFlowMapData_3D import ROLES, TEST_ROLES, measured_trajectories
from experiments.Build_Task678_FlowMap_1_1 import provenance
from experiments.Build_Task678_HanSampling_1_1 import DEFAULT_CONFIG
from experiments.Run_Task678_FlowMap_1_1 import truth_radius, interpolation


def matrix(spec):
    return [(d, arm, seed) for seed in spec['seeds'] for d in spec['datasets'] for arm in spec['neural_arms']]


def load_role(spec, dataset, role):
    root = Path(spec['output_root'])
    manifest = json.loads((root / 'build' / f'{dataset}.json').read_text())
    assert manifest['status'] == 'PASS' and manifest['config_sha256'] == spec['_config_sha256']
    records = [r for r in manifest['records'] if r['role'] == role]
    data, sources = [], []
    for record in records:
        path = root / 'cache' / dataset / Path(record['cache_file']).name
        assert record['config_sha256'] == spec['_config_sha256'] and sha256(path) == record['cache_sha256']
        with np.load(path, allow_pickle=False) as a:
            data.append({k: a[k] for k in a.files})
        sources.append({k: record[k] for k in ('ordinal', 'role', 'cache_sha256', 'regions', 'queries')})
    assert records
    return {k: np.concatenate([d[k] for d in data]) for k in data[0]}, sources


def derangement_by_window(data, seed):
    index = np.arange(len(data['origin0']))
    rng = np.random.default_rng(seed + 9101)
    for ordinal in np.unique(data['window_ordinal']):
        ids = np.flatnonzero(data['window_ordinal'] == ordinal)
        if len(ids) < 2:
            raise ValueError('Need two regions to test token necessity')
        cycle = rng.permutation(ids)
        index[cycle] = np.roll(cycle, 1)
    assert np.all(index != np.arange(len(index)))
    return index


def arrays_for(data, task, arm, permutation=None):
    if arm == 'coordinates_only':
        n = len(data['origin0'])
        features = dict(support0=np.zeros((n, 161), np.float32), support1=np.zeros((n, 161), np.float32),
                        context=np.zeros((n, 6, 161), np.float32))
    else:
        features = {k: data[f'{arm}__{k}'] for k in ('support0', 'support1', 'context')}
    if permutation is not None:
        features = {k: v[permutation] for k, v in features.items()}
    return training_arrays(data, features, task)


def composition_diagnostic(model, arrays, data, device):
    n = len(data['origin0'])
    second = {k: v[n:] for k, v in arrays.items()}
    steps = data['target0'].shape[2] - 1
    arrival = (data['target_long'][:, :, steps] - data['origin1'][:, None]) / data['radius1'][:, None, None]
    return predict_direct(model, second, device, query=arrival)


def save_result(directory, task, role, variant, prediction, data, sources, common, extra, diagnostic=None):
    prediction = np.asarray(prediction, np.float32)
    truth, radius = truth_radius(data, task)
    stem = f'{task}_{role}_{variant}'
    path = directory / f'{stem}_predictions.npz'
    values = {'prediction': prediction}
    if diagnostic is not None:
        values['true_arrival_diagnostic'] = np.asarray(diagnostic, np.float32)
        diagnostic_truth = data['target_long'][:, :, data['target0'].shape[2] - 1:]
        extra['true_arrival_diagnostic_metrics'] = measured_trajectories(values['true_arrival_diagnostic'], diagnostic_truth, data['radius1'])
        extra['true_arrival_diagnostic_scope'] = 'Separate diagnostic only; main Task8 always uses predicted arrival'
    np.savez_compressed(path, **values)
    metrics = measured_trajectories(prediction, truth, radius)
    windows = {}
    n = len(data['origin0'])
    for ordinal in np.unique(data['window_ordinal']):
        ix = np.flatnonzero(data['window_ordinal'] == ordinal)
        if task == 'Task6':
            ix = np.concatenate([ix, n + ix])
        windows[str(ordinal)] = measured_trajectories(prediction[ix], truth[ix], radius[ix])
    record = {**common, **extra, 'task': task, 'role': role, 'variant': variant, 'source_records': sources,
              'metrics': metrics, 'per_window': windows, 'prediction_file': path.name, 'prediction_sha256': sha256(path)}
    write_json(directory / f'{stem}.json', record)
    print(f"RESULT {common['dataset']}/{common['arm']}/{task}/{role}/{variant}: {metrics['position_nrmse']:.7g}", flush=True)


def run(spec, config, dataset, arm, seed, device='cuda'):
    spec['_config_sha256'] = sha256(config)
    directory = Path(spec['output_root']) / 'runs' / dataset / arm / f'seed{seed}'
    if directory.exists():
        raise FileExistsError('Do not overwrite or repeat an existing evaluation')
    directory.mkdir(parents=True)
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', '4')))
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Expected a GPU allocation')
    common = {**provenance(config), 'experiment': spec['experiment'], 'dataset': dataset, 'arm': arm, 'seed': seed,
        'device': torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU', 'settings': spec['decoder'],
        'frozen_encoder_sha256': sha256('FMT_Utils/DFT_FMT_3D.py'), 'checkpoint_files': 0,
        'token_bytes': 0 if arm == 'coordinates_only' else (644 if arm == 'fmt_all' else 2688),
        'physical_input_lines': 0 if arm == 'coordinates_only' else 7}
    assert common['frozen_encoder_sha256'] == spec['frozen_encoder_sha256']
    write_json(directory / 'started.json', common)
    try:
        models = {}
        if arm != 'affine':
            fit, fit_sources = load_role(spec, dataset, 'fit')
            for task in ('Task6', 'Task7'):
                def progress(item):
                    with (directory / f'{task}_learning_curve.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(item) + '\n'); f.flush()
                    print(f"FIT {dataset}/{arm}/{task} step={item['step']} seen_time_train={item['train_position_nrmse']:.6g}", flush=True)
                model, info = fit_dense(arrays_for(fit, task, arm), spec['decoder'], seed + int(task[-1]), device, progress)
                models[task] = model, info
                write_json(directory / f'{task}_training.json', {**common, **info, 'source_records': fit_sources})
            del fit
        count = 0
        for role in ROLES:
            data, sources = load_role(spec, dataset, role)
            variants = ['normal'] + (['shuffled_tokens'] if role in TEST_ROLES and arm in ('fmt_all', 'raw_positions') else [])
            for variant in variants:
                permutation = derangement_by_window(data, seed) if variant != 'normal' else None
                for task in ('Task6', 'Task7', 'Task8'):
                    info = {'parameter_count': 0, 'optimizer_steps': 0, 'input_dimension': 672} if arm == 'affine' else dict(models['Task6' if task == 'Task8' else task][1])
                    info.pop('learning_curve', None)
                    diagnostic = None
                    if arm == 'affine':
                        prediction = interpolation(data, task)
                    else:
                        model = models['Task6' if task == 'Task8' else task][0]
                        arrays = arrays_for(data, 'Task6' if task == 'Task8' else task, arm, permutation)
                        if task == 'Task8':
                            prediction, composition = compose_direct(model, arrays, device)
                            info.update(composition)
                            if variant == 'normal':
                                diagnostic = composition_diagnostic(model, arrays, data, device)
                        else:
                            prediction = predict_direct(model, arrays, device)
                    if task == 'Task8':
                        info.update(second_query_uses='predicted_first_stage_endpoint', training_source='Task6 short-map model or frozen affine interpolation')
                        if arm == 'affine':
                            steps = data['target0'].shape[2] - 1
                            arrival = (prediction[:, :, steps] - data['origin1'][:, None]) / data['radius1'][:, None, None]
                            info['predicted_arrival_outside_fraction'] = float((np.abs(arrival).sum(-1) > 1).mean())
                    if permutation is not None:
                        info['token_permutation'] = permutation.tolist()
                    save_result(directory, task, role, variant, prediction, data, sources, common, info, diagnostic)
                    count += 1
            del data
        write_json(directory / 'completed.json', {**common, 'status': 'PASS', 'metric_records': count, 'completed': provenance(config)})
    except Exception:
        write_json(directory / 'failed.json', {**common, 'status': 'FAILED', 'traceback': traceback.format_exc()})
        raise


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', default=DEFAULT_CONFIG); p.add_argument('--index', type=int, default=0); p.add_argument('--affine', action='store_true')
    args = p.parse_args(); spec = json.loads(Path(args.config).read_text())
    row = (spec['datasets'][args.index], 'affine', 0) if args.affine else matrix(spec)[args.index]
    run(spec, args.config, *row, device='cpu' if args.affine else 'cuda')


if __name__ == '__main__':
    main()
