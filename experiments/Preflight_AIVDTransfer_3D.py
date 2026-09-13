"""Training-only compatibility checks for the frozen AIVD transfer protocol."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from DeepUtils.utils import EasyConfig
from FMT_Utils.Task12Data_3D import _anchored_recipe
from FMT_Utils.DFT_FMT_3D import pathline_anchored_kinematic_dft_features_3d
from experiments.Verify_AIVDTransfer_3D import records, features, selected_ordinals
from experiments.Run_Task135_GeometricControls import write_json, sha
from experiments.Verify_HighReVAE import _train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/Verify_AIVDTransfer_1.1.json')
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    target = Path(spec['output_root'])/'preflight.json'
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(4)
    source = EasyConfig()
    source.update({'task2': {'batch_size': 256, 'weight_decay': 1e-5,
                             'learning_rate': 3e-4, 'target_optimizer_steps': 2}})
    rng = np.random.default_rng(7107)
    settings = {**spec['task2']['architecture'], 'optimizer_steps': 2}
    for dimension in (1, 29):
        x = rng.normal(size=(32, dimension)).astype(np.float32)
        result = _train(x, x[:8], settings, source, 100, torch.device('cpu'), return_model=True)
        assert result[2]['completed_optimizer_steps'] == 2
    checks = []
    for dataset in spec['datasets']:
        local = copy.deepcopy(spec)
        local.setdefault('dataset_splits', {}).setdefault(dataset, {}).setdefault('Task1', {})['train'] = [
            selected_ordinals(spec, 'Task1', dataset, 'train')[0]]
        rows = records(local, 'Task1', dataset, 'train')
        x = rows[0]['raw']
        values = {a: features(rows, spec['features'][a], 'cpu') for a in ('old_fmt', 'aivd', 'aivd_kin4')}
        for a, value in values.items():
            assert value.shape == (len(x), spec['feature_dimensions'][a])
            assert np.isfinite(value).all()
        direct = pathline_anchored_kinematic_dft_features_3d(
            torch.from_numpy(x), **_anchored_recipe('aivd1w3_dft')).astype(np.float32)
        np.testing.assert_array_equal(direct, values['aivd'])
        np.testing.assert_array_equal(values['aivd'], values['aivd_kin4'][:, :1])
        np.testing.assert_array_equal(values['old_fmt'][:, 161:], values['aivd_kin4'][:, 1:])
        checks.append({'dataset': dataset, 'ordinal': rows[0]['ordinal'],
                       'source_time': rows[0]['metadata']['source_time'], 'sample_count': len(x),
                       'dimensions': {a: v.shape[1] for a, v in values.items()},
                       'Task3_call_exact_match': True, 'kin4_retained_exactly': True})
        print(checks[-1], flush=True)
    write_json(target, {'status': 'PASS', 'config_sha256': sha(args.config),
                       'source_manifest_sha256': sha('SOURCE_MANIFEST.sha256'),
                       'checks': checks, 'end_time': time.time(),
                       'confirmation_opened': False, 'training_fixture_only': True,
                       'synthetic_VAE_updates_per_dimension': 2})


if __name__ == '__main__':
    main()
