"""Nested train-only expansion and batched physical labels for Task3+6 1.2."""
from pathlib import Path
import shutil

import numpy as np

from FMT_Utils.FlowMapData_3D import sha256, write_json
from FMT_Utils.Task36Labels_3D import labels_at_seeds, load_labels
from FMT_Utils.Task6Scarce_3D import subset_order
from experiments.Task6_DirectNeural_5_1 import assert_sources, read, provenance
from experiments.Task6_PrimitiveVAE_2_1 import strict_window


def attach_batch(spec, config, dataset, roles):
    """Read each source time slice once for all requested train seeds or test roles."""
    root = Path(spec['output_root'])
    source = Path(spec['source_cache']) / 'data' / dataset
    manifest = read(source / 'manifest.json')
    frozen = read(root / 'data' / dataset / 'manifest.json')
    assert sha256(source / 'manifest.json') == frozen['source_manifest_sha256']
    raw = Path(manifest['source']['source'])
    assert raw.stat().st_size == manifest['source']['source_bytes']
    assert raw.stat().st_mtime_ns == manifest['source']['source_mtime_ns']
    requests = {}
    for source_role in sorted({'train' if r.startswith('train_') else r for r in roles}):
        path = source / f'{source_role}.npz'
        assert sha256(path) == manifest['files'][source_role]['sha256']
        with np.load(path) as cache:
            origin, times, starts, ids = [cache[key] for key in ('origin', 'seed_time', 'source_start', 'primitive_id')]
            for role in roles:
                if ('train' if role.startswith('train_') else role) != source_role:
                    continue
                if role.startswith('train_'):
                    item = frozen['files'][role + '.npz']
                    assert sha256(item['path']) == item['sha256']
                    with np.load(item['path']) as train:
                        indices = train['source_indices'].copy()
                else:
                    indices = np.arange(len(origin), dtype=np.int64)
                requests[role] = dict(origin=origin[indices], time=times[indices], starts=starts[indices],
                    primitive_id=ids[indices], source_indices=indices, source_role=source_role,
                    labels=np.empty(len(indices), np.int64), ivd=np.empty(len(indices), np.float32),
                    threshold=np.empty(len(indices), np.float64), rows=[])
    all_starts = np.unique(np.concatenate([r['starts'] for r in requests.values()]))
    for start in all_starts:
        field, meta = strict_window(raw, int(start), 2, spec['label_max_spatial_dim'])
        selected = [(role, r['starts'] == start) for role, r in requests.items() if (r['starts'] == start).any()]
        origins = np.concatenate([requests[role]['origin'][mask] for role, mask in selected])
        labels, values, threshold = labels_at_seeds(field, origins)
        offset = 0
        for role, mask in selected:
            request = requests[role]
            count = int(mask.sum())
            np.testing.assert_allclose(request['time'][mask], meta['source_time'], rtol=0, atol=1e-6)
            request['labels'][mask] = labels[offset:offset+count]
            request['ivd'][mask] = values[offset:offset+count]
            request['threshold'][mask] = threshold
            request['rows'].append(dict(source_start=int(start), source_time=meta['source_time'], samples=count,
                positive=int(labels[offset:offset+count].sum()), threshold=threshold, spatial_strides=meta['spatial_strides']))
            offset += count
        print(dict(dataset=dataset, labels_source_start=int(start), samples=len(origins)), flush=True)
    folder = root / 'labels' / dataset
    folder.mkdir(parents=True, exist_ok=True)
    for role, request in requests.items():
        assert set(np.unique(request['labels'])) == {0, 1}, f'{dataset}/{role}: one class'
        path = folder / f'{role}.npz'
        assert not path.exists()
        np.savez(path, **{key: request[key] for key in ('labels', 'ivd', 'threshold', 'source_indices', 'primitive_id')})
        write_json(folder / f'{role}.json', dict(provenance=provenance(config), dataset=dataset, role=role,
            samples=len(request['labels']), positive=int(request['labels'].sum()), rows=request['rows'],
            source_role_sha256=manifest['files'][request['source_role']]['sha256'], labels_sha256=sha256(path),
            source_manifest_sha256=sha256(source / 'manifest.json'), labels_are_inputs=False,
            definition='IVD(seed,t0) >= whole loaded field p95, frozen 1.1 operator'))


def prepare(spec, config, index):
    assert_sources(spec)
    previous = Path(spec['previous_cache'])
    assert sha256(previous / 'config.frozen.json') == spec['previous_config_sha256']
    dataset = spec['datasets'][index]
    source = Path(spec['source_cache']) / 'data' / dataset
    subset = Path(spec['subset_cache']) / 'data' / dataset
    manifest, small = read(source / 'manifest.json'), read(subset / 'manifest.json')
    assert read(source / 'audit.json')['passed']
    assert small['source_manifest_sha256'] == sha256(source / 'manifest.json')
    assert small['provenance']['config_sha256'] == spec['subset_config_sha256']
    assert small['source_train_sha256'] == manifest['files']['train']['sha256']
    assert sha256(source / 'train.npz') == manifest['files']['train']['sha256']
    folder = Path(spec['output_root']) / 'data' / dataset
    folder.mkdir(parents=True, exist_ok=False)
    files = {}
    with np.load(source / 'train.npz') as cache:
        geometry, identities, scales = [cache[key] for key in ('geometry', 'primitive_id', 'scale_id')]
        for seed in [spec['search_seed']] + spec['seeds']:
            name = f'train_{seed}.npz'
            assert sha256(subset / name) == small['files'][name]
            ids = subset_order(len(geometry), seed)[:max(spec['train_sizes'])]
            assert len(ids) == max(spec['train_sizes']) == len(np.unique(ids))
            with np.load(subset / name) as old:
                np.testing.assert_array_equal(ids[:len(old['source_indices'])], old['source_indices'])
                np.testing.assert_array_equal(geometry[ids[:len(old['geometry'])]], old['geometry'])
            target = folder / name
            np.savez(target, geometry=geometry[ids], source_indices=ids, primitive_id=identities[ids], scale_id=scales[ids])
            files[name] = dict(path=str(target.resolve()), sha256=sha256(target))
    name = 'validation.npz'
    assert sha256(subset / name) == small['files'][name]
    assert small['source_validation_sha256'] == manifest['files']['validation']['sha256']
    files[name] = dict(path=str((subset / name).resolve()), sha256=sha256(subset / name))
    write_json(folder / 'manifest.json', dict(provenance=provenance(config), dataset=dataset, files=files,
        source_manifest_sha256=sha256(source / 'manifest.json'), subset_manifest_sha256=sha256(subset / 'manifest.json'),
        train_sizes=spec['train_sizes'], test_read=False, policy='Same frozen subset_order extended on train only'))
    roles = [f'train_{seed}' for seed in [spec['search_seed']] + spec['seeds']]
    attach_batch(spec, config, dataset, roles)
    # Check recalculated labels against all old training prefixes, then reuse the
    # unchanged validation labels. No source validation field re-read is needed.
    label_folder = Path(spec['output_root']) / 'labels' / dataset
    for role in roles + ['validation']:
        old_meta = read(previous / 'labels' / dataset / f'{role}.json')
        old_path = previous / 'labels' / dataset / f'{role}.npz'
        assert old_meta['provenance']['config_sha256'] == spec['previous_config_sha256']
        assert old_meta['source_manifest_sha256'] == sha256(source / 'manifest.json')
        assert sha256(old_path) == old_meta['labels_sha256']
        source_role = 'train' if role.startswith('train_') else role
        assert old_meta['source_role_sha256'] == manifest['files'][source_role]['sha256']
        if role == 'validation':
            assert not (label_folder / f'{role}.npz').exists()
            shutil.copyfile(old_path, label_folder / f'{role}.npz')
            write_json(label_folder / f'{role}.json', dict(old_meta, provenance=provenance(config),
                previous_metadata_sha256=sha256(previous / 'labels' / dataset / f'{role}.json')))
        else:
            with np.load(old_path) as old, np.load(label_folder / f'{role}.npz') as new:
                for key in ('labels', 'primitive_id', 'source_indices'):
                    np.testing.assert_array_equal(new[key][:len(old[key])], old[key])


def prepare_test(spec, config, index):
    root = Path(spec['output_root'])
    assert sha256(root / 'selection.json') == sha256(root / 'selection.before_test.json')
    selection = read(root / 'selection.json')
    assert selection['provenance']['config_sha256'] == sha256(config) and not selection['test_read']
    attach_batch(spec, config, spec['datasets'][index], ['test', 'unseen_scale'])
