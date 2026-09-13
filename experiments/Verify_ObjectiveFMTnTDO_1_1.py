"""Small real-data verification of the separate nTDO 1.1 module."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from FMT_Utils import objective_fmt_nTDO as module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discrepancy(a, b):
    delta = np.asarray(a)-np.asarray(b)
    return {'max_absolute': float(np.max(np.abs(delta))),
            'relative_l2': float(np.linalg.norm(delta)/max(np.linalg.norm(a), 1e-30))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/Verify_ObjectiveFMTnTDO_1.1.json')
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    assert spec['module_version'] == module.VERSION
    output = Path(spec['output_root']); output.mkdir(parents=True, exist_ok=False)
    rows = []
    torch.set_num_threads(4)
    frozen_path = Path(module.__file__).with_name('DFT_FMT_3D.py')
    frozen_hash = sha(frozen_path)
    with threadpool_limits(limits=4):
        for dataset in spec['datasets']:
            source = Path(spec['source_root'])/dataset/'geometry/geometry_bundle.npz'
            with np.load(source, allow_pickle=False) as z:
                x = np.asarray(z['paths'][:spec['samples_per_dataset']], dtype=spec['dtype'])
            assert len(x) == spec['samples_per_dataset']
            t = np.linspace(0., 1., x.shape[2])
            angle = spec['transform']['angle_over_window_radians']*t
            q = np.zeros((len(t), 3, 3))
            q[:, 0, 0] = q[:, 1, 1] = np.cos(angle)
            q[:, 0, 1] = -np.sin(angle); q[:, 1, 0] = np.sin(angle); q[:, 2, 2] = 1.
            c = np.stack((t, t*t, np.sin(t)), axis=-1)
            y = np.einsum('tij,nktj->nkti', q, x) + c[None, None]
            a, b = module.build_fourier_inputs(x), module.build_fourier_inputs(y)
            expected = np.einsum('tij,nktj->nkti', q, a['relative'].numpy())
            np.testing.assert_allclose(b['relative'], expected, atol=1e-11, rtol=1e-10)
            np.testing.assert_allclose(a['distances'], b['distances'], atol=1e-11, rtol=1e-10)
            before = module.objective_fmt_nTDO(x, spec['num_freq'], return_blocks=True)
            after = module.objective_fmt_nTDO(y, spec['num_freq'], return_blocks=True)
            for key, width in spec['expected_widths'].items():
                assert before[key].shape == (len(x), width)
            np.testing.assert_allclose(before['distance_fourier'], after['distance_fourier'], atol=1e-10, rtol=1e-10)
            row = {'dataset': dataset, 'source': str(source), 'source_sha256': sha(source),
                   'n': len(x), 'vector_input_transformation_residual': discrepancy(expected, b['relative']),
                   'distance_input': discrepancy(a['distances'], b['distances']),
                   'output_errors': {key: discrepancy(before[key], after[key]) for key in before},
                   'fourier_input_shapes': {key: list(a[key].shape) for key in ('relative', 'distances')}}
            rows.append(row)
            print(json.dumps(row), flush=True)
    assert sha(frozen_path) == frozen_hash
    result = {'experiment': spec['experiment'], 'execution_status': 'PASS',
              'module_version': module.VERSION, 'config_sha256': sha(args.config),
              'runner_sha256': sha(__file__), 'module_sha256': sha(module.__file__),
              'tests_sha256': sha('tests/test_objective_fmt_ntdo.py'),
              'frozen_encoder_sha256_unchanged': frozen_hash,
              'numpy': np.__version__, 'torch': torch.__version__,
              'distance_block_observer_test': 'PASS',
              'combined_features_invariance_claimed': False,
              'labels_read': False, 'models_trained': 0, 'rows': rows}
    (output/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('PASS: declared input arrays and real-data distance block verified.', flush=True)


if __name__ == '__main__':
    main()
