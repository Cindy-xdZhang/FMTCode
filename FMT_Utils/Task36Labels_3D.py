"""Attach whole-field IVD-p95 labels to the unchanged Task6 material seeds."""
import json
from pathlib import Path

import numpy as np

from FMT_Utils.FlowMapData_3D import sha256,write_json
from FMT_Utils.FMT_3D_pipeline import compute_ivd_reference_3d
from experiments.Task6_PrimitiveVAE_2_1 import strict_window
from experiments.Task6_DirectNeural_5_1 import prepare as prepare_flow,read,provenance


def labels_at_seeds(field,origins):
    # Keep the frozen scalar and interpolation operator. Never fill missing fluid
    # with zero or silently change the percentile domain.
    if not np.isfinite(field.field[0]).all():
        raise ValueError('IVD source frame has missing velocities; no labels fabricated')
    volume,values,_=compute_ivd_reference_3d(field,0.,origins)
    assert np.isfinite(volume).all() and np.isfinite(values).all()
    threshold=float(np.percentile(volume,95.))
    return (values.astype(np.float64)>=threshold).astype(np.int64),values,threshold


def attach_labels(spec,config,dataset,role):
    root=Path(spec['output_root'])
    source=Path(spec['source_cache'])/'data'/dataset
    manifest=read(source/'manifest.json')
    frozen=read(root/'data'/dataset/'manifest.json')
    assert sha256(source/'manifest.json')==frozen['source_manifest_sha256']
    train_role=role.startswith('train_')
    source_role='train' if train_role else role
    source_path=source/f'{source_role}.npz'
    assert sha256(source_path)==manifest['files'][source_role]['sha256']
    if train_role:
        item=frozen['files'][f'{role}.npz']
        assert sha256(item['path'])==item['sha256']
        with np.load(item['path']) as data:
            ids=data['source_indices'][:spec['primary_train_size']].copy()
    else:
        ids=None
    with np.load(source_path) as data:
        if ids is None:
            ids=np.arange(len(data['origin']),dtype=np.int64)
        origins=data['origin'][ids].copy()
        times=data['seed_time'][ids].copy()
        starts=data['source_start'][ids].copy()
        primitive_ids=data['primitive_id'][ids].copy()
    raw_source=Path(manifest['source']['source'])
    stat=raw_source.stat()
    assert stat.st_size==manifest['source']['source_bytes']
    assert stat.st_mtime_ns==manifest['source']['source_mtime_ns']
    folder=root/'labels'/dataset
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{role}.npz'
    assert not path.exists()
    labels=np.empty(len(ids),np.int64)
    ivd=np.empty(len(ids),np.float32)
    thresholds=np.empty(len(ids),np.float64)
    rows=[]
    for start in np.unique(starts):
        mask=starts==start
        field,meta=strict_window(raw_source,int(start),2,spec['label_max_spatial_dim'])
        np.testing.assert_allclose(times[mask],meta['source_time'],rtol=0,atol=1e-6)
        lab,values,threshold=labels_at_seeds(field,origins[mask])
        labels[mask],ivd[mask],thresholds[mask]=lab,values,threshold
        rows.append(dict(source_start=int(start),source_time=meta['source_time'],samples=int(mask.sum()),
            positive=int(lab.sum()),threshold=threshold,spatial_strides=meta['spatial_strides']))
    assert set(np.unique(labels))=={0,1},f'{dataset}/{role}: both classes are required; samples unchanged'
    np.savez(path,labels=labels,ivd=ivd,threshold=thresholds,source_indices=ids,primitive_id=primitive_ids)
    write_json(folder/f'{role}.json',dict(provenance=provenance(config),dataset=dataset,role=role,
        samples=len(labels),positive=int(labels.sum()),rows=rows,source_role_sha256=sha256(source_path),
        labels_sha256=sha256(path),source_manifest_sha256=sha256(source/'manifest.json'),
        definition='IVD(seed,t0) >= percentile95(whole loaded field IVD at t0)',
        labels_are_inputs=False))


def load_labels(spec,config,dataset,role):
    folder=Path(spec['output_root'])/'labels'/dataset
    meta=read(folder/f'{role}.json')
    assert meta['provenance']['config_sha256']==sha256(config)
    assert sha256(folder/f'{role}.npz')==meta['labels_sha256']
    with np.load(folder/f'{role}.npz') as data:
        assert np.array_equal(data['labels'],(data['ivd']>=data['threshold']).astype(np.int64))
        return data['labels'].copy()


def prepare(spec,config,index):
    dataset=spec['datasets'][index]
    prepare_flow(spec,config,index)
    for seed in [spec['search_seed']]+spec['seeds']:
        attach_labels(spec,config,dataset,f'train_{seed}')
    attach_labels(spec,config,dataset,'validation')


def prepare_test(spec,config,index):
    dataset=spec['datasets'][index]
    root=Path(spec['output_root'])
    assert sha256(root/'selection.json')==sha256(root/'selection.before_test.json')
    selection=read(root/'selection.json')
    assert selection['provenance']['config_sha256']==sha256(config) and selection['gate_passed']
    for role in ('test','unseen_scale'):
        attach_labels(spec,config,dataset,role)
