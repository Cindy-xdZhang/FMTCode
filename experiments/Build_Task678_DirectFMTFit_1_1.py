"""Keep old held-out windows; add genuinely new material seeds within training windows."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from FMT_Utils.FlowMapData_3D import build_window, sha256, write_json
from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d
from experiments.Build_Task678_FlowMap_1_1 import make_features, separated_seed_grid, provenance, check_present_frames

DEFAULT_CONFIG = 'config/Verify_Task678_DirectFMTFit_1.1.json'
DATA_KEYS = ('target0', 'target1', 'target_long', 'origin0', 'origin1', 'radius0', 'radius1', 'context_origins', 'duration', 'material_bundle_ids')


def compact(data, features=None):
    feature = make_features(data, ['fmt_all']) if features is None else features
    return {**{k: data[k] for k in DATA_KEYS}, **{f'fmt__{k}': feature[f'fmt_all__{k}'] for k in ('support0', 'support1', 'context')}}


def jittered_centers(lo, hi, radius, grid_side, count, seed):
    grid, shape, separation = separated_seed_grid(lo, hi, radius, grid_side)
    if separation <= 10*radius:
        raise ValueError('Training replicate must retain within-replicate context separation')
    amplitude = min(.4*radius, (separation-10*radius)/(4*np.sqrt(3)))
    rng = np.random.default_rng(seed)
    centers = grid[rng.permutation(len(grid))[:count]]
    centers += rng.uniform(-amplitude, amplitude, centers.shape)
    return centers, dict(seed_grid_shape=shape, base_grid_separation=separation,
                         jitter_amplitude=amplitude, minimum_separation_bound=separation-2*np.sqrt(3)*amplitude)


def save_cache(path, data, metadata):
    if path.exists():
        saved=json.loads(path.with_suffix('.json').read_text())
        assert saved['config_sha256']==metadata['config_sha256'] and saved['cache_sha256']==sha256(path)
        assert saved['base_cache_sha256']==metadata['base_cache_sha256'] and saved['replicate']==metadata['replicate']
        # The expanded builder reuses exact original caches made by prepare.
        return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    metadata = {**metadata, 'cache_sha256': sha256(path), 'cache_file': str(path), 'regions': len(data['origin0'])}
    write_json(path.with_suffix('.json'), metadata)
    return metadata


def build(spec, config, dataset, base_only=False):
    root = Path(spec['output_root'])
    old = Path(spec['base_output_root']) / 'cache' / dataset
    if sha256('FMT_Utils/DFT_FMT_3D.py') != spec['frozen_encoder_sha256']:
        raise ValueError('Frozen FMT source has changed')
    manifest=root/'build'/f"{dataset}{'_base' if base_only else ''}.json"
    if manifest.exists():
        raise FileExistsError('Already built; inspect before resubmitting')
    base = json.loads(Path(spec['base_config']).read_text())
    if sha256(spec['base_config']) != spec['base_config_sha256']:
        raise ValueError('Base protocol changed')
    records = []
    frame_sets = {}
    for ordinal in range(8):
        p = old/f'window_{ordinal:02d}.npz'
        meta = json.loads(p.with_suffix('.json').read_text())
        if meta['config_sha256'] != spec['base_config_sha256'] or sha256(p) != meta['cache_sha256']:
            raise ValueError('Base cache identity mismatch')
        frame_sets[ordinal] = set(range(meta['start_index'], meta['end_index']+1))
        assert not any(frame_sets[ordinal] & frame_sets[k] for k in range(ordinal))
        role = next(k for k,v in base['splits'].items() if ordinal in v)
        assert meta['role'] == role
        if dataset in base['cylinder_datasets']:
            assert meta['time_start'] >= 7.5-1e-7
        fp = old/f'window_{ordinal:02d}_features.npz'
        fm = json.loads(fp.with_suffix('.json').read_text())
        assert fm['config_sha256'] == spec['base_config_sha256']
        assert fm['cache_sha256'] == meta['cache_sha256'] and fm['feature_sha256'] == sha256(fp)
        with np.load(p, allow_pickle=False) as a:
            data = {k:a[k] for k in a.files if k != 'metadata_json'}
        with np.load(fp, allow_pickle=False) as a:
            feat = {f'fmt_all__{k}':a[f'fmt_all__{k}'] for k in ('support0','support1','context')}
        common = {**provenance(config), 'dataset':dataset, 'ordinal':ordinal, 'role':role,
            'start_index':meta['start_index'], 'end_index':meta['end_index'],
            'time_start':meta['time_start'], 'time_end':meta['time_end'],
            'base_cache_sha256':meta['cache_sha256'], 'base_feature_sha256':fm['feature_sha256']}
        records.append(save_cache(root/'cache'/dataset/f'window_{ordinal:02d}_rep00.npz', compact(data,feat),
                                  {**common,'replicate':0,'construction':'exact original material data and native FMT'}))
        if role != 'train' or base_only:
            continue
        check_present_frames(meta['source_path'], meta['start_index'],9)
        field, field_meta = load_netcdf_window_3d(meta['source_path'],meta['start_index'],9,base['sampling']['max_spatial_dim'])
        assert hashlib.sha256(np.ascontiguousarray(field.field).tobytes()).hexdigest() == meta['source_window_sha256']
        radius = float(data['radius0'][0])
        lo = field.domainMinBoundary.astype(float)+5.5*radius
        hi = field.domainMaxBoundary.astype(float)-5.5*radius
        for repeat in range(1,spec['training_replicates']):
            seed = spec['data_seed']+ordinal*100+repeat
            centers, grid_meta = jittered_centers(lo,hi,radius,base['sampling']['seed_grid_side'],base['sampling']['max_bundles'],seed)
            extra, integration = build_window(field,centers,radius,field.tmax/2,62,seed)
            assert len(extra['origin0']) >= base['sampling']['minimum_bundles']
            path = root/'cache'/dataset/f'window_{ordinal:02d}_rep{repeat:02d}.npz'
            records.append(save_cache(path,compact(extra),{**common,**grid_meta,**integration,**field_meta,
                'replicate':repeat,'construction':'new jittered material supports and new independently integrated query particles',
                'source_window_sha256':meta['source_window_sha256']}))
            print(f"BUILT {dataset}/{ordinal}/{repeat}: {len(extra['origin0'])} regions",flush=True)
    counts = {role:sum(r['regions'] for r in records if r['role']==role) for role in ('train','validation','test')}
    original = sum(r['regions'] for r in records if r['role']=='train' and r['replicate']==0)
    write_json(manifest,{**provenance(config),'status':'PASS','dataset':dataset,'records':records,
        'counts':counts,'base_training_regions':original,'training_expansion_ratio':counts['train']/original,
        'overlap_policy':'Training replicas can overlap one another spatially; held-out physical time windows remain disjoint'})
    print(json.dumps({'dataset':dataset,'counts':counts,'training_expansion_ratio':counts['train']/original}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=DEFAULT_CONFIG);p.add_argument('--index',type=int,default=0);p.add_argument('--base-only',action='store_true')
    a=p.parse_args();spec=json.loads(Path(a.config).read_text());build(spec,a.config,spec['datasets'][a.index],a.base_only)

if __name__=='__main__':
    main()
