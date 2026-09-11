"""Read-only, independently implemented audit of frozen Task6 5.1 artifacts."""
import argparse
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import socket
import subprocess

import numpy as np
import torch

from FMT_Utils.Task6DirectNeural_3D import direct_inputs, fit_statistics, DirectGeometryVAE, audit_direct_model


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def independent_errors(prediction, truth):
    """Per-primitive squared physical errors; no production metric function."""
    p, y = np.asarray(prediction, np.float64), np.asarray(truth, np.float64)
    assert p.shape == y.shape and p.shape[1:] == (7, 32, 3)
    assert np.isfinite(p).all() and np.isfinite(y).all()
    d = p-y
    squared = np.einsum('nltc,nltc->nlt', d, d)
    pair = np.zeros(len(y), np.float64)
    for i, j in itertools.combinations(range(7), 2):
        predicted_distance = np.sqrt(np.sum((p[:, i, 1:]-p[:, j, 1:])**2, axis=2))
        true_distance = np.sqrt(np.sum((y[:, i, 1:]-y[:, j, 1:])**2, axis=2))
        pair += np.mean((predicted_distance-true_distance)**2, axis=1)/21.
    return dict(position_rmse_r=squared[:, :, 1:].mean((1, 2)),
        all_time_rmse_r=squared.mean((1, 2)), initial_rmse_r=squared[:, :, 0].mean(1),
        center_rmse_r=squared[:, 0, 1:].mean(1), neighbor_rmse_r=squared[:, 1:, 1:].mean((1, 2)),
        endpoint_rmse_r=squared[:, :, -1].mean(1), pair_distance_rmse_r=pair,
        time_rmse_r=squared.mean(1))


def aggregate_errors(errors, mask):
    result = {key: np.sqrt(value[mask].mean(0)).tolist() for key, value in errors.items()}
    return dict(result, samples=int(np.sum(mask)))


def inverse_for_audit_only(features):
    """Diagnostic only. This function is never used in training or prediction."""
    f = np.asarray(features, np.float64)
    coefficients = f[:, :336].reshape(-1, 7, 16, 3).astype(np.complex128)
    coefficients[:, :, 1:] += 1j*f[:, 336:].reshape(-1, 7, 15, 3)
    increments = np.fft.irfft(coefficients, n=31, axis=2, norm='forward')
    initial = np.concatenate([np.zeros((1, 3)), np.eye(3).repeat(2, 0)*np.tile([1., -1.], 3)[:, None]])
    paths = np.zeros((len(f), 7, 32, 3), np.float64)
    paths[:, :, 1:] = np.cumsum(increments, axis=2)
    paths[:, 1:] += paths[:, :1]+initial[None, 1:, None]
    return paths


def feature_diagnostic(y, validation):
    result = {}
    states = []
    for arm in ('raw_neural', 'signed_fmt16_neural'):
        x, vx = direct_inputs(y, arm), direct_inputs(validation, arm)
        stats = fit_statistics(x, y)
        q, vq = (x-stats['input_mean'])/stats['input_std'], (vx-stats['input_mean'])/stats['input_std']
        eigen = np.linalg.eigvalsh(q.T@q/len(q)).clip(0.)[::-1]
        fraction = eigen/eigen.sum()
        positive = fraction > 0
        raw_std = x.std(0, dtype=np.float64)
        result[arm] = dict(input_std_quantiles=np.quantile(raw_std, [0, .1, .5, .9, 1]).tolist(),
            replaced_near_constant_channels=int(np.sum(raw_std < 1e-6)),
            variance_rank_99=int(np.searchsorted(np.cumsum(fraction), .99)+1),
            effective_covariance_rank=float(np.exp(-np.sum(fraction[positive]*np.log(fraction[positive])))),
            train_standardized_rms=float(np.sqrt(np.mean(q*q))), validation_standardized_rms=float(np.sqrt(np.mean(vq*vq))),
            train_standardized_max=float(np.abs(q).max()), validation_standardized_max=float(np.abs(vq).max()),
            validation_fraction_outside_training_channel_range=float(np.mean((vq < q.min(0)) | (vq > q.max(0)))))
        torch.manual_seed(17)
        model = DirectGeometryVAE(stats)
        states.append([p.detach().clone() for p in model.parameters()])
        result[arm]['structure'] = audit_direct_model(model)
        if arm == 'signed_fmt16_neural':
            for name, geometry, features in [('train', y, x), ('validation', validation, vx)]:
                delta = inverse_for_audit_only(features)[:, :, 1:]-geometry[:, :, 1:]
                rmse = float(np.sqrt(np.mean(np.sum(delta*delta, axis=-1))))
                magnitude = float(np.sqrt(np.mean(np.sum(geometry[:, :, 1:].astype(float)**2, axis=-1))))
                result[arm][name+'_analytic_roundtrip'] = dict(rmse_r=rmse, max_coordinate_error=float(np.abs(delta).max()), relative_rmse=rmse/max(magnitude, 1e-12))
                assert rmse/max(magnitude, 1e-12) < 1e-6
    assert all(torch.equal(a, b) for a, b in zip(*states))
    result['paired_random_parameters_equal'] = True
    result['note'] = 'Analytic inverse used only here to audit information preservation, never in frozen neural inference. Feature statistics are descriptive, not a causal ablation.'
    return result


def audit_flow(config, index):
    audit_spec = read(config)
    run = Path(audit_spec['source_run'])
    spec = read(run/'config.frozen.json')
    assert digest(run/'config.frozen.json') == audit_spec['source_config_sha256']
    for prefix in ('source', 'subset'):
        assert digest(Path(spec[prefix+'_cache'])/'config.frozen.json') == spec[prefix+'_config_sha256']
    checkout = run.parents[1]
    for name, expected in audit_spec['scientific_file_sha256'].items():
        assert digest(checkout/name) == expected
    selection_hash = digest(run/'selection.json')
    assert selection_hash == digest(run/'selection.before_test.json') == audit_spec['selection_sha256']
    dataset = spec['datasets'][index]
    source = Path(spec['source_cache'])/'data'/dataset
    subset = Path(spec['subset_cache'])/'data'/dataset
    manifest = read(source/'manifest.json')
    current = read(run/'data'/dataset/'manifest.json')
    assert current['source_manifest_sha256'] == digest(source/'manifest.json')
    assert current['subset_manifest_sha256'] == digest(subset/'manifest.json')
    data, frames, identities, bounds = {}, {}, {}, {}
    for role in ('train', 'validation', 'test', 'unseen_scale'):
        path = source/f'{role}.npz'
        assert digest(path) == manifest['files'][role]['sha256']
        with np.load(path) as f:
            data[role] = dict(geometry=f['geometry'], scale_id=f['scale_id'])
            identities[role] = set(f['primitive_id'].tolist())
            pairs = set(zip(f['source_start'].tolist(), f['source_end'].tolist()))
            frames[role] = {i for a, b in pairs for i in range(int(a), int(b)+1)}
            times = f['seed_time']
            low, high = manifest['source']['plan']['seed_time_bounds']
            assert np.all((times >= low-1e-6) & (times <= high+1e-6))
            if dataset.startswith(('cylinder', 'halfcylinder')):
                assert low == 7.5 and high == 12.
            bounds[role] = dict(samples=len(times), first_seed_time=float(times.min()), last_seed_time=float(times.max()))
    for a, b in itertools.combinations(data, 2):
        assert not identities[a].intersection(identities[b])
        if {a, b} != {'test', 'unseen_scale'}:
            assert not frames[a].intersection(frames[b])
    cached_subsets = {}
    with np.load(source/'train.npz') as full:
        for seed in [spec['search_seed']]+spec['seeds']:
            name = f'train_{seed}.npz'
            assert digest(subset/name) == current['files'][name]['sha256']
            with np.load(subset/name) as f:
                indices = f['source_indices']
                np.testing.assert_array_equal(indices, np.random.default_rng(seed).permutation(len(data['train']['geometry']))[:4096])
                np.testing.assert_array_equal(f['primitive_id'], full['primitive_id'][indices])
                np.testing.assert_array_equal(f['geometry'], data['train']['geometry'][indices])
                cached_subsets[seed] = f['geometry']
    assert digest(subset/'validation.npz') == current['files']['validation.npz']['sha256']
    with np.load(subset/'validation.npz') as f:
        np.testing.assert_array_equal(f['geometry'], data['validation']['geometry'])
    del data['train']
    diagnostic = feature_diagnostic(cached_subsets[spec['search_seed']][:1024], data['validation']['geometry'])
    compared_rows, model_count, failures, max_error = 0, 0, [], 0.
    prediction_hashes = {}
    selection = read(run/'selection.json')
    for n in spec['train_sizes']:
        for seed in spec['seeds']:
            folder = run/'final'/dataset/str(n)/str(seed)
            result = read(folder/'result.json')
            assert result['selection_sha256'] == selection_hash
            assert result['provenance']['git_commit'] == audit_spec['scientific_commit']
            assert result['provenance']['config_sha256'] == audit_spec['source_config_sha256']
            assert result['provenance']['time'] > selection['provenance']['time']
            assigned = [r for f in result['fits']+result['failures'] for r in f['roles']]
            assert sorted(assigned) == ['fmt', 'pca', 'raw_matched', 'raw_selected']
            for name in ('signed_fmt16_neural_c2', 'raw_neural_c2', 'raw_neural_c1'):
                fit = read(folder/name/'fit.json')
                assert fit['updates'] == 20000 and fit['train_samples'] == n and fit['statistics_train_samples'] == n
                assert fit['structure']['trainable_parameters'] == 4267990
                best = min((r for r in fit['curve'] if r['step'] > 0), key=lambda r: r['validation_rmse_r'])
                assert fit['selected_step'] == best['step'] and fit['validation_rmse_r'] == best['validation_rmse_r']
                y = cached_subsets[seed][:n].astype(float)[:, :, 1:]
                scale = float(np.sqrt(np.mean(np.sum((y-y.mean(0))**2, axis=-1))))
                np.testing.assert_allclose(scale, fit['target_scale'], rtol=1e-7)
                actual_failed = (folder/name/'validation_failed.json').exists()
                assert actual_failed == (fit['validation_rmse_r'] >= spec['validation_gate_rmse_r'])
            for failure in result['failures']:
                assert not list((folder/failure['name']).glob('*_predictions.npz'))
                failures.append(dict(dataset=dataset, train_size=n, seed=seed, **failure))
            for fit in result['fits']:
                model_count += 1
                for role in ('test', 'unseen_scale'):
                    path = folder/fit['name']/f'{role}_predictions.npz'
                    prediction_hashes[str(path.relative_to(run))] = digest(path)
                    with np.load(path) as f:
                        per_sample = independent_errors(f['prediction'], data[role]['geometry'])
                    scales = data[role]['scale_id']
                    records = [m for m in fit['metrics'] if m['role'] == role]
                    assert sorted(r['scale_id'] for r in records) == [-1]+sorted(np.unique(scales).tolist())
                    for r in records:
                        mask = np.ones(len(scales), bool) if r['scale_id'] == -1 else scales == r['scale_id']
                        fresh = aggregate_errors(per_sample, mask)
                        for key, value in fresh.items():
                            np.testing.assert_allclose(value, r[key], rtol=1e-10, atol=1e-12)
                            max_error = max(max_error, float(np.max(np.abs(np.asarray(value)-np.asarray(r[key])))))
                        compared_rows += len(fit['roles'])
    assert not any(p for suffix in ('*.pt', '*.pth', '*.ckpt') for p in (run/'final'/dataset).rglob(suffix))
    result = dict(dataset=dataset, independent_audit_passed=True, experiment_passed=not failures,
        models_with_predictions=model_count, compared_rows=compared_rows, failures=failures,
        max_metric_absolute_difference=max_error, data_bounds=bounds, data_frames_disjoint=True,
        subset_matches_original_geometry=True, diagnostic=diagnostic, prediction_sha256=prediction_hashes,
        provenance=dict(config_sha256=digest(config), scientific_commit=audit_spec['scientific_commit'],
            source_config_sha256=audit_spec['source_config_sha256'], selection_sha256=selection_hash,
            node=socket.gethostname(), job_id=os.getenv('SLURM_JOB_ID'), array_id=os.getenv('SLURM_ARRAY_TASK_ID'),
            completed=datetime.datetime.now().astimezone().isoformat()))
    write(Path(audit_spec['output_root'])/f'{dataset}.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('diagnostic', 'prediction_sha256', 'failures')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/Verify_Task6_DirectAudit_1.1.json')
    p.add_argument('--index', type=int, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    audit_flow(args.config, args.index)
