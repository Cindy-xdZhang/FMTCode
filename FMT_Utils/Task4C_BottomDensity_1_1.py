"""Append bottom-region training bundles to the frozen 4.14 dataset."""
from __future__ import annotations
import copy
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from experiments import Task4C_PhysicalLength_4_14 as base
from experiments import Task4C_Multiscale_4_1 as storage
from FMT_Utils.Task4C_PaperBundles_3_1 import load_flow, cell_centers
from FMT_Utils.Task4C_HairpinBinary_2_1 import read_dataset, sample_gt


def read_metadata(folder):
    with np.load(Path(folder)/'metadata.npz') as z:
        return {key:z[key] for key in z.files}


def evaluation_clearance(center, tree, centers, scales):
    near=tree.query_ball_point(center,float(scales.max()))
    return not near or bool(np.all(np.linalg.norm(centers[near]-center,axis=1)>=scales[near]))


def generate(spec,index,source,pilot=False,input_root=None):
    flow=spec['flows'][index];name=flow['name'];rule=spec['density']
    original=read_metadata(source/'train');evaluation=[read_metadata(source/s) for s in ('validation','test')]
    centers=np.concatenate([m['center'] for m in evaluation]);scales=np.concatenate([m['local_grid_scale'] for m in evaluation])
    tree=cKDTree(centers);original_centers=cKDTree(original['center'])
    root=Path(input_root or spec['input_root'])
    for key in ('flow','gt'):
        assert base.pipeline.sha(root/flow[key])==flow[key+'_sha256']
    axes,_,heads,oyf,grid,_=load_flow(root/flow['flow'],flow['lambda2_threshold'])
    gt=read_dataset(root/flow['gt']);components,catalog,details=base.build_catalog(axes,heads,gt,spec)
    details['labels']=[{k:h[k] for k in ('head_component','label','instance')} for h in catalog]
    instances=sorted(np.unique(original['instance'][original['labels']==1]).tolist())
    assert len(instances)==rule['expected_original_training_instances'][name]
    quotas=rule['pilot_bundles_per_instance'] if pilot else rule['additional_bundles_per_existing_training_instance']
    plan=base.scale_plan(spec,name);all_rows=[];records=[]
    for instance in instances:
        candidates=[];z_all=[]
        for head in catalog:
            if head['label']!=1 or head['instance']!=instance:continue
            ids=head['split_cells']['train'];points=cell_centers(ids,components.shape,axes)
            if not len(ids):continue
            item=dict(head,cell_ids=ids);candidates.append(item);z_all.extend(points[:,2].tolist())
        assert candidates,f'No original positive training cells for instance {instance}'
        z_cut=float(np.quantile(z_all,rule['bottom_cell_fraction']))
        pool=[]
        for head in candidates:
            points=cell_centers(head['cell_ids'],components.shape,axes)
            lower=dict(head,cell_ids=head['cell_ids'][points[:,2]<=z_cut])
            if len(lower['cell_ids']):pool.append(lower)
        accepted=[];pending={k:[] for k in range(len(plan))};attempts=0;rejected={}
        def reject(reason):rejected[reason]=rejected.get(reason,0)+1
        def flush():
            for sid,batch in pending.items():
                if not batch:continue
                good,bad,_=base.trace_batch(grid,batch,plan[sid],spec)
                accepted.extend(good[:max(0,quotas-len(accepted))])
                for item in bad:reject(item['reason'])
                batch.clear()
        # Cycle the original nine scale tuples; failed integration never changes a label.
        for attempt in range(rule['maximum_center_attempts_per_instance']):
            sid=attempt%len(plan);round_id=attempt//len(plan)
            # Retain all original candidate cells. Every second center is biased
            # toward lower cells; thin heads are not excluded by a new hard cut.
            available=pool if round_id%2 else candidates
            head=available[(round_id//2)%len(available)]
            number=rule['center_number_start']+round_id
            sample=base.center_and_neighbors(head,components,axes,oyf,number,plan[sid],spec['sampling']['seed']+index*100000)
            attempts+=1
            if sample is None or len(sample['seeds'])<10:reject('fewer_than_10_head_seeds');continue
            if not evaluation_clearance(sample['center'],tree,centers,scales):reject('original_evaluation_clearance');continue
            if original_centers.query(sample['center'])[0]<=1e-12:reject('original_center_duplicate');continue
            sample.update(head_component=head['head_component'],instance=instance,label=1,scale_id=sid,
                center_number=number,local_grid_scale=sample['neighbor_distance']/plan[sid]['neighbor_grid_scale'],
                nearest_train_center_distance=0.,nearest_same_head_train_center_distance=0.)
            pending[sid].append(sample)
            if sum(map(len,pending.values()))>=36:flush()
            if len(accepted)>=quotas:break
        flush()
        if len(accepted)!=quotas:
            raise ValueError(f'{name} instance {instance}: {len(accepted)}/{quotas}; rejections={rejected}; no label/distance/length relaxation')
        all_rows.extend(accepted)
        records.append(dict(instance=int(instance),added=len(accepted),bottom_cell_z_cut=z_cut,attempts=attempts,rejections=rejected,
            scale_counts=np.bincount([r['scale_id'] for r in accepted],minlength=len(plan)).tolist()))
        print(json.dumps(dict(flow=name,instance=int(instance),added=len(accepted),instances_done=len(records),instances_total=len(instances))),flush=True)
    return all_rows,records,details


def save_added(rows,dest,spec):
    extra=('half_arc_lengths','half_step_counts','ds','maxiteration','requested_half_length','local_grid_scale',
        'nearest_train_center_distance','nearest_same_head_train_center_distance')
    values={key:np.asarray([r[key] for r in rows]) for key in extra}
    settings=copy.deepcopy(spec);settings['sampling']['minimum_class_count']=0
    storage.save_split(rows,dest,settings)
    metadata=read_metadata(dest);metadata.update(values)
    np.savez_compressed(dest/'metadata.npz',**metadata)


def prepare(spec,config,index,identity,write,sha,pilot=False,input_root=None,source_root=None):
    name=spec['flows'][index]['name'];src=Path(source_root or spec['source_output'])/'physical'/name
    old=json.loads((src/'preparation.json').read_text())
    assert old['identity']['commit']==spec['source_scientific_commit']
    source_hashes={}
    for split,entry in old['splits'].items():
        for filename,digest in entry['files'].items():
            assert sha(src/split/filename)==digest
        source_hashes[split]=entry['files']
    root=Path(spec['output'])/('pilot' if pilot else '')/'physical'/name
    root.mkdir(parents=True,exist_ok=False)
    rows,records,catalog=generate(spec,index,src,pilot,input_root)
    added=root/'added_train';save_added(rows,added,spec)
    report=dict(complete=True,identity=identity(config),source_output=str(src),source_commit=spec['source_scientific_commit'],
        source_hashes=source_hashes,instances=records,original_catalog=catalog,scale_plan=base.scale_plan(spec,name),splits={},pilot=pilot)
    if pilot:
        write(root/'preparation.json',report);return
    for split in ('train','validation','test'):
        dest=root/split;dest.mkdir()
        if split!='train':
            for filename in ('geometry.npy','seeds.npy','metadata.npz'):shutil.copyfile(src/split/filename,dest/filename)
        else:
            for filename in ('geometry.npy','seeds.npy'):
                original=np.load(src/split/filename,mmap_mode='r');extra=np.load(added/filename,mmap_mode='r')
                joined=np.lib.format.open_memmap(dest/filename,mode='w+',dtype=original.dtype,shape=(len(original)+len(extra),*original.shape[1:]))
                joined[:len(original)]=original;joined[len(original):]=extra;joined.flush();del joined
            original=read_metadata(src/split);extra=read_metadata(added)
            assert original.keys()==extra.keys()
            np.savez_compressed(dest/'metadata.npz',**{k:np.concatenate((original[k],extra[k])) for k in original})
        meta=read_metadata(dest)
        report['splits'][split]=dict(samples=len(meta['labels']),class_counts=np.bincount(meta['labels'],minlength=2).tolist(),
            files={f:sha(dest/f) for f in ('geometry.npy','seeds.npy','metadata.npz')})
    shutil.copyfile(src/'cell_split.npz',root/'cell_split.npz')
    report['original_rows_retained']=True
    write(root/'preparation.json',report)
