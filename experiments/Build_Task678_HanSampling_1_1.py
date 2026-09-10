"""Build dense Sobol queries with particle, primitive and time-window holdouts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d
from FMT_Utils.FlowMapData_3D import sha256, write_json
from FMT_Utils.HanFlowMapData_3D import dense_window, support_features, partition_roles
from experiments.Build_Task678_FlowMap_1_1 import inspect_source, check_present_frames, separated_seed_grid, provenance

DEFAULT_CONFIG = 'config/Verify_Task678_HanSampling_1.1.json'


def save_roles(root, dataset, data, row, settings, metadata):
    data = {**data, **support_features(data)}
    data['window_ordinal'] = np.full(len(data['origin0']), row['ordinal'], np.int64)
    records = []
    for role, part in partition_roles(data, row['role'], settings).items():
        if len(part['origin0']) < settings['minimum_role_bundles']:
            raise ValueError(f'Too few retained {role} regions')
        path = Path(root) / 'cache' / dataset / f"window_{row['ordinal']:02d}_{role}.npz"
        if path.exists():
            raise FileExistsError(f'Never replace prior data: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **part)
        record = {**metadata, 'ordinal': row['ordinal'], 'role': role, 'source_role': row['role'],
                  'cache_file': str(path), 'cache_sha256': sha256(path), 'regions': len(part['origin0']),
                  'queries': part['target0'].shape[1], 'material_bundle_ids': part['material_bundle_ids'].tolist()}
        write_json(path.with_suffix('.json'), record)
        records.append(record)
    return records


def build(spec, config, dataset):
    root, settings = Path(spec['output_root']), spec['sampling']
    if (root / 'build' / f'{dataset}.json').exists():
        raise FileExistsError('Completed data build already exists')
    source = inspect_source(spec, dataset)
    records = []
    write_json(root / 'build' / f'{dataset}.started.json', {**provenance(config), **source})
    for row in source['windows']:
        check_present_frames(source['source'], row['start_index'], settings['source_frames'])
        field, field_meta = load_netcdf_window_3d(source['source'], row['start_index'],
            settings['source_frames'], settings['max_spatial_dim'])
        if not np.isfinite(field.field).all():
            raise ValueError('Nonfinite source field')
        change = float(np.linalg.norm(np.diff(field.field.astype(float), axis=0)))
        if change <= 1e-12:
            raise ValueError('This experiment requires genuinely unsteady fields')
        radius = float(np.min(field.gridInterval)) * settings['radius_grid_fraction']
        lo, hi = field.domainMinBoundary + 5.5 * radius, field.domainMaxBoundary - 5.5 * radius
        grid, shape, separation = separated_seed_grid(lo, hi, radius, settings['seed_grid_side'])
        if np.any(lo >= hi) or separation <= 10 * radius:
            raise ValueError('Insufficient separated spatial regions')
        rng = np.random.default_rng(settings['data_seed'] + row['ordinal'])
        centers = grid[rng.permutation(len(grid))[:settings['max_bundles']]]
        data, audit = dense_window(field, centers, radius, field.tmax / 2, settings,
                                    settings['data_seed'] + 100000 * row['ordinal'])
        if len(data['origin0']) < settings['minimum_bundles']:
            raise ValueError(f'Insufficient common coverage: {audit}')
        metadata = {**provenance(config), **field_meta, **row, **audit, 'dataset': dataset,
            'source_window_sha256': hashlib.sha256(np.ascontiguousarray(field.field).tobytes()).hexdigest(),
            'temporal_change_l2': change, 'grid_shape': shape, 'grid_min_center_separation': separation,
            'status': 'PASS'}
        records.extend(save_roles(root, dataset, data, row, settings, metadata))
        print(f"BUILT {dataset}/{row['ordinal']}: {len(data['origin0'])}/{len(centers)} regions, dense independent queries", flush=True)
    counts = {role: sum(r['regions'] for r in records if r['role'] == role) for role in {r['role'] for r in records}}
    write_json(root / 'build' / f'{dataset}.json', {**provenance(config), 'dataset': dataset,
        'status': 'PASS', 'records': records, 'counts': counts, 'source': source})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', default=DEFAULT_CONFIG); p.add_argument('--index', type=int, default=0)
    args = p.parse_args(); spec = json.loads(Path(args.config).read_text())
    build(spec, args.config, spec['datasets'][args.index])


if __name__ == '__main__':
    main()
