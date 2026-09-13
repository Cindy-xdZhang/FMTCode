"""Versioned Task1/Task2 transfer of the unchanged Task3 AIVD feature."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from DeepUtils.utils import EasyConfig
from FMT_Utils.Task12Data_3D import feature_matrix
from FMT_Utils.Task12Evaluation_3D import (
    fit_kmeans_transform, calibrate_vortex_cluster, binary_cluster_metrics,
)
from experiments.Verify_FMTAllV2_3D import (
    evidence, features, labels, prepare_task2, encode_test,
)
from experiments.Run_Task135_GeometricControls import load_records, sha, write_json, write_csv
from experiments.Verify_HighReVAE import _train


def selected_ordinals(spec, task, dataset, role):
    return spec.get('dataset_splits', {}).get(dataset, {}).get(task, {}).get(
        role, spec[task.lower()][role])


def records(spec, task, dataset, role):
    phase = 'confirmation' if role == 'confirmation' and task == 'Task1' else 'development'
    result = load_records(spec, 'Task1', dataset, phase,
                          selected_ordinals(spec, task, dataset, role))
    if dataset in spec['cylinder_datasets']:
        if any(float(r['metadata']['source_time']) < spec['minimum_cylinder_time'] for r in result):
            raise ValueError('early cylinder slice in registered late-time experiment')
    return result


def feature_diagnostics(rows, device):
    """Inspect the original slice context without changing it or dropping samples."""
    result = []
    for r in rows:
        x = r['raw']
        tensor = torch.from_numpy(x).to(device)
        d = torch.stack((tensor[:, 1]-tensor[:, 2], tensor[:, 3]-tensor[:, 4],
                         tensor[:, 5]-tensor[:, 6]), dim=-1)
        singular = torch.linalg.svdvals(d[:, :3])
        ratio = singular[..., -1] / singular[..., 0].clamp_min(1e-30)
        value = features([r], 'aivd1w3_dft', device)
        result.append({
            'path': r['path'], 'ordinal': r['ordinal'], 'sample_count': len(x),
            'source_time': r['metadata']['source_time'],
            'mean_context': 'all_retained_primitives_in_this_source_slice',
            'input_dimension': value.shape[1],
            'minimum': float(value.min()), 'maximum': float(value.max()),
            'mean': float(value.mean()), 'std': float(value.std()),
            'near_rank_cutoff_fraction_first_three_samples': float((ratio <= 1e-6).float().mean()),
            'physical_sample_offsets_first_four': r['times'][0, :4].tolist(),
            'shared_physical_times_within_slice': bool(np.all(r['times'] == r['times'][:1])),
            'derivative_clock': 'sample_index_unchanged_from_Task3',
        })
    return result


def run(spec, config_path, task, dataset, seed):
    part = spec[task.lower()]
    if dataset not in spec['datasets'] or seed not in part['seeds']:
        raise ValueError('unregistered run')
    target = Path(spec['output_root'])/'shards'/task/dataset/f'seed{seed}'
    if target.exists():
        raise FileExistsError(f'refusing to overwrite {target}')
    target.mkdir(parents=True)
    device = torch.device('cpu' if task == 'Task1' else 'cuda')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    started = {
        'experiment': spec['experiment'], 'task': task, 'dataset': dataset, 'seed': seed,
        'config_sha256': sha(config_path), 'runner_sha256': sha(__file__),
        'encoder_sha256': sha(Path(__file__).resolve().parents[1]/'FMT_Utils/DFT_FMT_3D.py'),
        'source_manifest_sha256': sha('SOURCE_MANIFEST.sha256'),
        'base_commit': spec['base_commit'],
        'job': os.environ.get('SLURM_JOB_ID'), 'array_task': os.environ.get('SLURM_ARRAY_TASK_ID'),
        'node': socket.gethostname(), 'start_time': time.time(),
        'device': torch.cuda.get_device_name(0) if device.type == 'cuda' and torch.cuda.is_available() else 'CPU',
        'torch': torch.__version__, 'numpy': np.__version__,
    }
    write_json(target/'started.json', started)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('allocated GPU required')
    train = records(spec, task, dataset, 'train')
    validation = records(spec, task, dataset, 'validation')
    train_y, val_y = labels(train), labels(validation)
    write_json(target/'input_audit.json', {
        'training': evidence(train), 'validation': evidence(validation),
        'feature_diagnostics': feature_diagnostics(train, device),
    })
    models, frozen = {}, {}
    if task == 'Task1':
        for arm in part['arms']:
            name = spec['features'][arm]
            x, y = features(train, name, device), features(validation, name, device)
            # A scalar needs no dimensionality reduction; all higher-dimensional
            # arms retain the frozen PCA8. This is fixed before any evaluation.
            pca_dim = None if x.shape[1] == 1 else part['pca_dim']
            model = fit_kmeans_transform(x, pca_dim, seed, part['kmeans_n_init'])
            cluster = calibrate_vortex_cluster(val_y, model.predict(y))
            models[arm] = (model, cluster, name)
            frozen[arm] = {
                'feature': name, 'vortex_cluster': cluster, 'input_dimension': x.shape[1],
                'pca_dimension': pca_dim,
                'calibration': binary_cluster_metrics(val_y, model.predict(y), cluster),
            }
    else:
        source = EasyConfig()
        source.update({'task2': {
            'batch_size': part['batch_size'], 'weight_decay': part['weight_decay'],
            'learning_rate': part['architecture']['learning_rate'],
            'target_optimizer_steps': part['architecture']['optimizer_steps'],
        }})
        for arm in part['arms']:
            x, y, transform = prepare_task2(train, validation, arm,
                spec['features']['old_fmt'], device, spec['features'][arm])
            train_mu, val_mu, losses, model = _train(
                x, y, part['architecture'], source, seed, device, return_model=True)
            clusterer = KMeans(n_clusters=2, random_state=part['kmeans_seed'],
                              n_init=part['kmeans_n_init']).fit(train_mu)
            cluster = calibrate_vortex_cluster(val_y, clusterer.predict(val_mu))
            models[arm] = (model, clusterer, cluster, transform)
            frozen[arm] = {
                'feature': spec['features'][arm], 'vortex_cluster': cluster,
                'input_dimension': x.shape[1], 'losses': losses,
                'calibration': binary_cluster_metrics(val_y, clusterer.predict(val_mu), cluster),
            }
            print(f'TRAINED {task} {dataset} seed={seed} arm={arm}', flush=True)
    write_json(target/'frozen_models.json', frozen)
    # No test slice is opened until every arm and cluster mapping is fixed.
    frozen_at = time.time()
    del train, validation
    gc.collect()
    test = records(spec, task, dataset, 'confirmation')
    write_json(target/'confirmation_input_audit.json', evidence(test))
    test_y = labels(test)
    result_rows, predictions = [], {}
    for arm, item in models.items():
        if task == 'Task1':
            model, cluster, name = item
            cluster_ids = model.predict(features(test, name, device))
        else:
            model, clusterer, cluster, transform = item
            cluster_ids = clusterer.predict(encode_test(test, transform, model, device))
        outcome = binary_cluster_metrics(test_y, cluster_ids, cluster)
        predictions[arm] = (cluster_ids == cluster).astype(np.uint8)
        result_rows.append({'task': task, 'dataset': dataset, 'seed': seed, 'arm': arm,
                            'sample_count': len(test_y), **outcome})
    np.savez_compressed(target/'predictions.npz', labels=test_y,
                        **{f'prediction_{a}': v for a, v in predictions.items()})
    write_csv(target/'per_run.csv', result_rows)
    write_json(target/'complete.json', {
        **started, 'status': 'COMPLETE', 'end_time': time.time(),
        'models_frozen_before_test_time': frozen_at,
        'rows': len(result_rows), 'per_run_sha256': sha(target/'per_run.csv'),
        'prediction_sha256': sha(target/'predictions.npz'), 'checkpoints_written': 0,
    })
    print(json.dumps(result_rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/Verify_AIVDTransfer_1.1.json')
    parser.add_argument('--task', choices=['Task1', 'Task2'], required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    seeds = spec[args.task.lower()]['seeds']
    dataset_index, seed_index = divmod(args.index, len(seeds))
    run(spec, args.config, args.task, spec['datasets'][dataset_index], seeds[seed_index])


if __name__ == '__main__':
    main()
