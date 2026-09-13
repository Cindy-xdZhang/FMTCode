"""Direct observer test of the exact center-masked frozen FMT encoder."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from FMT_Utils import DFT_FMT_3D as frozen
from FMT_Utils.Task12Data_3D import feature_matrix


TRANSFORMS = ('identity', 'time_translation', 'constant_rotation',
              'time_rotation', 'time_rotation_translation')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(x, name):
    t = np.linspace(0., 1., x.shape[2])
    q = np.broadcast_to(np.eye(3), (len(t), 3, 3)).copy()
    if name == 'constant_rotation':
        # Exact proper signed permutation: no trigonometric round-off.
        q[:] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    elif name in ('time_rotation', 'time_rotation_translation'):
        a = 1.3 * t
        q[:, 0, 0] = np.cos(a); q[:, 0, 1] = -np.sin(a)
        q[:, 1, 0] = np.sin(a); q[:, 1, 1] = np.cos(a)
    c = np.zeros((len(t), 3))
    if name in ('time_translation', 'time_rotation_translation'):
        c = np.stack((t, t*t, np.sin(t)), axis=1)
    np.testing.assert_allclose(q @ q.transpose(0, 2, 1),
                               np.broadcast_to(np.eye(3), q.shape), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(q), 1., atol=1e-14)
    # One common Q(t), c(t) for all material particles. No re-seeding.
    return np.einsum('tij,nktj->nkti', q, x) + c[None, None], q, c


def scalars(x):
    i, j = np.triu_indices(7, 1)
    distances = np.linalg.norm(x[:, i] - x[:, j], axis=-1)
    d = (x[:, 1:] - x[:, :1]).transpose(0, 2, 1, 3)
    return distances, d @ d.swapaxes(-1, -2)


def encode(x):
    old = frozen.pathline_dft_features_3d(
        torch.from_numpy(x), num_freq=6, neighbor_scale=1., neighbor_weight=1.,
        neighbor_pool='sort', mode='gram', include_chirality=True)
    assert old.shape == (len(x), 161)
    masked = old.copy()
    masked[:, :23] = 0.
    np.testing.assert_array_equal(masked[:, 23:], old[:, 23:])
    kin = feature_matrix({'raw': x.reshape(len(x), -1), 'fmt': old,
                          'features': {}}, 'kin4', 'cpu')
    assert kin.shape == (len(x), 28)
    return {'neighbor_only': old[:, 23:], 'no_center_core': masked,
            'kin4_only': kin, 'no_center_with_kin4': np.concatenate((masked, kin), axis=1)}


def error(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    delta = np.abs(a-b)
    bad = delta > (1e-5 + 1e-5*np.abs(a))
    flat = np.unravel_index(np.argmax(delta), delta.shape)
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {'max_absolute': float(delta.max()), 'l2_difference': float(np.linalg.norm(a-b)),
            'symmetric_relative_l2': float(np.linalg.norm(a-b)/max(norm_a, norm_b, 1e-30)),
            'base_l2': norm_a, 'transformed_l2': norm_b,
            'changed_primitives': int(np.any(bad.reshape(len(a), -1), axis=1).sum()),
            'sample_count': len(a), 'allclose_rtol_atol_1e_5': bool(not bad.any()),
            'max_index': list(map(int, flat)), 'base_at_max': float(a[flat]),
            'transformed_at_max': float(b[flat])}


def evaluate(name, source, x, dtype, out, save_fixture=False):
    original = x.astype(dtype)
    base = encode(original)
    distances, gram = scalars(original.astype(np.float64))
    rows = []
    for observer in TRANSFORMS:
        y, q, c = transform(x.astype(np.float64), observer)
        y = y.astype(dtype)
        after = encode(y)
        d, g = scalars(y.astype(np.float64))
        geometric = {'pair_distance': error(distances, d), 'same_time_gram': error(gram, g)}
        expected = np.einsum('tij,nktj->nkti', q, original[:, 1:]-original[:, :1])
        geometric['relative_vector_transformation'] = error(expected, y[:, 1:]-y[:, :1])
        row = {'dataset': name, 'source': source, 'dtype': np.dtype(dtype).name,
               'transform': observer, 'n': len(x), 'geometry': geometric,
               'features': {key: error(base[key], after[key]) for key in base}}
        if observer == 'identity':
            for key in base:
                np.testing.assert_array_equal(base[key], after[key])
        if dtype == np.float64:
            np.testing.assert_allclose(distances, d, rtol=1e-10, atol=1e-11)
        rows.append(row)
        if save_fixture and dtype == np.float64 and observer == 'time_rotation':
            np.savez_compressed(out/'stationary_cross_counterexample.npz',
                                original=original, transformed=y, rotation=q, translation=c,
                                original_distance=distances, transformed_distance=d,
                                original_feature=base['no_center_core'],
                                transformed_feature=after['no_center_core'])
    print(json.dumps({'dataset': name, 'dtype': np.dtype(dtype).name,
                      'rotation_neighbor_error': rows[3]['features']['neighbor_only'],
                      'rotation_distance_error': rows[3]['geometry']['pair_distance']}), flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='C:/Users/xingdi/sources/FLowlineClusteringVisualAnalysis/outputs/Other_FeatureSilhouette_1.1')
    p.add_argument('--output', default='outputs/Verify_FMTNoCenterObjectivity_1.1')
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    encoder_hash = sha(frozen.__file__)
    offsets = np.vstack((np.zeros(3), np.eye(3)[0], -np.eye(3)[0],
                         np.eye(3)[1], -np.eye(3)[1], np.eye(3)[2], -np.eye(3)[2]))
    stationary = np.broadcast_to(offsets[None, :, None, :], (1, 7, 32, 3)).copy()
    files = sorted(Path(args.root).glob('*/geometry/geometry_bundle.npz'))
    assert len(files) == 8
    sources, rows = [], []
    with threadpool_limits(limits=4):
        for dtype in (np.float64, np.float32):
            rows.extend(evaluate('stationary_unit_cross', None, stationary, dtype, out, True))
        for file in files:
            with np.load(file, allow_pickle=False) as z:
                x = np.asarray(z['paths'], dtype=np.float64)
            sources.append({'path': str(file), 'sha256': sha(file), 'n': len(x)})
            for dtype in (np.float64, np.float32):
                rows.extend(evaluate(file.parent.parent.name, str(file), x, dtype, out))
    assert encoder_hash == sha(frozen.__file__)
    # This is a counterexample test, not a pass certificate for the encoder.
    fixture = next(r for r in rows if r['dataset']=='stationary_unit_cross'
                   and r['dtype']=='float64' and r['transform']=='time_rotation')
    assert fixture['features']['no_center_core']['base_l2'] == 0.
    assert fixture['features']['no_center_core']['max_absolute'] > .1
    result = {'experiment': 'Verify_FMTNoCenterObjectivity_1.1',
              'execution_status': 'PASS', 'encoder_observer_invariance': 'FAIL',
              'definition': 'Frozen encoder, scale=weight=1, only first23 center slots zeroed; kin4 audited separately.',
              'source_sha256': sha(__file__), 'frozen_encoder_sha256': encoder_hash,
              'auxiliary_adapter_sha256': sha(Path(frozen.__file__).with_name('Task12Data_3D.py')),
              'numpy': np.__version__, 'torch': torch.__version__,
              'test_labels_used': False, 'trained_models': 0, 'source_files': sources, 'rows': rows}
    (out/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    flat_rows = [{'dataset': r['dataset'], 'dtype': r['dtype'], 'transform': r['transform'],
                  'block': key, **{k:v for k,v in value.items() if k!='max_index'}}
                 for r in rows for key, value in r['features'].items()]
    with (out/'feature_errors.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(flat_rows[0])); w.writeheader(); w.writerows(flat_rows)
    print('Completed: verification executed successfully; no-center encoder invariance fails.', flush=True)


if __name__ == '__main__':
    main()
