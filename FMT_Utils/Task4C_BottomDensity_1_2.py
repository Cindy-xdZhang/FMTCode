"""Fivefold training data through fresh seeds with frozen sampling strata."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from FMT_Utils import Task4C_BottomDensity_1_1 as previous
from experiments import Task4C_PhysicalLength_4_14 as base

read_metadata=previous.read_metadata


def lower_pool_flags(metadata,spec):
    original_count=spec['sampling']['per_flow_counts']['train']
    is_added=np.arange(len(metadata['labels']))>=original_count
    assert np.all(metadata['labels'][is_added]==1)
    assert np.all(metadata['center_number'][is_added]>=spec['density']['center_number_start'])
    return is_added & ((metadata['center_number']-spec['density']['center_number_start'])%2==1)


def candidate_pools(catalog,components,axes,metadata,spec):
    """Use exactly the full/lower cell pools from 1.1; do not add a GT crop."""
    pools={(r['head_component'],False):dict(r,cell_ids=r['split_cells']['train']) for r in catalog}
    positive=np.unique(metadata['instance'][metadata['labels']==1])
    for instance in positive:
        candidates=[r for r in catalog if r['label']==1 and r['instance']==instance and len(r['split_cells']['train'])]
        heights=np.concatenate([previous.cell_centers(r['split_cells']['train'],components.shape,axes)[:,2] for r in candidates])
        cut=float(np.quantile(heights,spec['density']['bottom_cell_fraction']))
        for r in candidates:
            cells=r['split_cells']['train'];z=previous.cell_centers(cells,components.shape,axes)[:,2]
            pools[(r['head_component'],True)]=dict(r,cell_ids=cells[z<=cut])
    return pools


def generate(spec,index,source,pilot=False,input_root=None):
    flow=spec['flows'][index];name=flow['name'];rule=spec['expansion'];root=Path(input_root or spec['input_root'])
    metadata=read_metadata(source/'train');n=len(metadata['labels']);lower=lower_pool_flags(metadata,spec)
    assert n==rule['source_rows_per_flow'][name]
    evaluation=[read_metadata(source/s) for s in ('validation','test')]
    eval_centers=np.concatenate([m['center'] for m in evaluation]);eval_h=np.concatenate([m['local_grid_scale'] for m in evaluation])
    eval_tree=cKDTree(eval_centers);source_tree=cKDTree(metadata['center'])
    for key in ('flow','gt'):assert base.pipeline.sha(root/flow[key])==flow[key+'_sha256']
    axes,_,heads,oyf,grid,_=previous.load_flow(root/flow['flow'],flow['lambda2_threshold'])
    components,catalog,details=base.build_catalog(axes,heads,previous.read_dataset(root/flow['gt']),spec)
    pools=candidate_pools(catalog,components,axes,metadata,spec);plan=base.scale_plan(spec,name)
    strata=np.column_stack((metadata['head_component'],metadata['scale_id'],lower.astype(np.int32)))
    selected=np.sort(np.unique(strata,axis=0,return_index=True)[1]) if pilot else np.arange(n)
    replicas=range(1 if pilot else rule['new_centers_per_source_bundle'])
    stride=n*rule['new_centers_per_source_bundle'];all_rows=[];rejections={};trace_calls=0
    assert rule['new_center_number_start']+rule['maximum_attempts_per_new_bundle']*stride<np.iinfo(np.int32).max
    for replica in replicas:
        for first in range(0,len(selected),rule['integration_batch_templates']):
            ids=selected[first:first+rule['integration_batch_templates']];waiting=ids.tolist();accepted={};attempts={int(i):0 for i in ids}
            def reject(reason):rejections[reason]=rejections.get(reason,0)+1
            while waiting:
                pending={sid:[] for sid in range(len(plan))}
                for row_id in waiting:
                    attempt=attempts[row_id];attempts[row_id]+=1
                    if attempt>=rule['maximum_attempts_per_new_bundle']:
                        raise ValueError(f'{name}/row{row_id}/replica{replica} exhausted retries: {rejections}; no rule relaxation')
                    sid=int(metadata['scale_id'][row_id]);head=pools[(int(metadata['head_component'][row_id]),bool(lower[row_id]))]
                    assert len(head['cell_ids']) and head['label']==metadata['labels'][row_id] and head['instance']==metadata['instance'][row_id]
                    number=rule['new_center_number_start']+row_id*rule['new_centers_per_source_bundle']+replica+attempt*stride
                    sample=base.center_and_neighbors(head,components,axes,oyf,number,plan[sid],spec['sampling']['seed']+index*100000)
                    if sample is None or len(sample['seeds'])<10:reject('fewer_than_10_head_seeds');continue
                    if not previous.evaluation_clearance(sample['center'],eval_tree,eval_centers,eval_h):reject('evaluation_clearance');continue
                    if source_tree.query(sample['center'])[0]<=1e-12:reject('old_center_duplicate');continue
                    sample.update(head_component=head['head_component'],instance=head['instance'],label=head['label'],scale_id=sid,
                        center_number=number,local_grid_scale=sample['neighbor_distance']/plan[sid]['neighbor_grid_scale'],
                        nearest_train_center_distance=0.,nearest_same_head_train_center_distance=0.,source_row=row_id,replica=replica)
                    pending[sid].append(sample)
                for sid,samples in pending.items():
                    if not samples:continue
                    good,bad,_=base.trace_batch(grid,samples,plan[sid],spec);trace_calls+=1
                    for row in good:accepted[row['source_row']]=row
                    for row in bad:reject(row['reason'])
                waiting=[i for i in waiting if i not in accepted]
            all_rows.extend(accepted[int(i)] for i in ids)
            print(json.dumps(dict(flow=name,replica=replica,source_rows_done=first+len(ids),source_rows_total=len(selected),added=len(all_rows))),flush=True)
    provenance=dict(source_row=np.array([r['source_row'] for r in all_rows],np.int32),
        replica=np.array([r['replica'] for r in all_rows],np.int8),lower_pool=lower[[r['source_row'] for r in all_rows]])
    details['labels']=[{key:r[key] for key in ('head_component','label','instance')} for r in catalog]
    return all_rows,provenance,dict(catalog=details,rejections=rejections,trace_calls=trace_calls,source_strata=len(np.unique(strata,axis=0)))


def prepare(spec,config,index,identity,write,sha,pilot=False,input_root=None,source_root=None):
    name=spec['flows'][index]['name'];source=Path(source_root or spec['source_output'])/'physical'/name
    old=json.loads((source/'preparation.json').read_text());assert old['identity']['git_commit']==spec['source_scientific_commit']
    source_hashes={}
    for split,entry in old['splits'].items():
        for filename,digest in entry['files'].items():assert sha(source/split/filename)==digest
        source_hashes[split]=entry['files']
    dest=Path(spec['output'])/('pilot' if pilot else '')/'physical'/name;dest.mkdir(parents=True,exist_ok=False)
    rows,provenance,details=generate(spec,index,source,pilot,input_root)
    added=dest/'added_train';previous.save_added(rows,added,spec)
    np.savez_compressed(dest/'augmentation_index.npz',**provenance)
    report=dict(complete=True,pilot=pilot,identity=identity(config),source_output=str(source),source_hashes=source_hashes,
        source_commit=spec['source_scientific_commit'],generation=details,scale_plan=base.scale_plan(spec,name),splits={},
        augmentation_index_sha256=sha(dest/'augmentation_index.npz'))
    if pilot:write(dest/'preparation.json',report);return
    for split in ('train','validation','test'):
        target=dest/split;target.mkdir()
        if split!='train':
            for filename in ('geometry.npy','seeds.npy','metadata.npz'):shutil.copyfile(source/split/filename,target/filename)
        else:
            for filename in ('geometry.npy','seeds.npy'):
                old_array=np.load(source/split/filename,mmap_mode='r');new_array=np.load(added/filename,mmap_mode='r')
                combined=np.lib.format.open_memmap(target/filename,mode='w+',dtype=old_array.dtype,shape=(len(old_array)+len(new_array),*old_array.shape[1:]))
                combined[:len(old_array)]=old_array;combined[len(old_array):]=new_array;combined.flush();del combined
            old_meta=read_metadata(source/split);new_meta=read_metadata(added);assert old_meta.keys()==new_meta.keys()
            np.savez_compressed(target/'metadata.npz',**{key:np.concatenate((old_meta[key],new_meta[key])) for key in old_meta})
        m=read_metadata(target)
        report['splits'][split]=dict(samples=len(m['labels']),class_counts=np.bincount(m['labels'],minlength=2).tolist(),
            files={filename:sha(target/filename) for filename in ('geometry.npy','seeds.npy','metadata.npz')})
    shutil.copyfile(source/'cell_split.npz',dest/'cell_split.npz')
    write(dest/'preparation.json',report)
