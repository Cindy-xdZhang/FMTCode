"""Task4-c local generalization at newly integrated centers near training sites."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import Task4C_ManifoldMixup_4_4 as pipeline
from experiments import Task4C_Multiscale_4_1 as physical
from FMT_Utils.Task4C_LinePooling_4_3 import make_model, line_fmt, bundle_voxels
from FMT_Utils.Task4C_Multiscale_4_1 import transform_fmt

DEFAULT_CONFIG = 'config/mainExp_Task4C_NearbySampling_4.13.json'


def load_spec(config):
    definition = json.loads(Path(config).read_text())
    if pipeline.sha(definition['base_config']) != definition['base_config_sha256']:
        raise ValueError('Frozen base config changed')
    if 'flows' in definition:
        return definition
    base = json.loads(Path(definition['base_config']).read_text())
    spec = {k:base[k] for k in ('flows','input_root','bundles','sampling','scale_sets',
        'cache_source','parent_output','parent_config_sha256','encoding')}
    spec.update(definition)
    return spec


def identity(config):
    result = pipeline.identity(config)
    for name in ('experiments/Task4C_NearbySampling_4_13.py','ibex_bash/task4c_nearby_sampling_4p13.sh'):
        result['source_sha256'][name] = pipeline.sha(name)
    d = json.loads(Path(config).read_text())
    result['source_sha256'][d['base_config']] = pipeline.sha(d['base_config'])
    return result


def anchor_plan(n,counts,seed):
    if sum(counts.values()) > n:
        raise ValueError('Nearby splits require distinct original anchor rows')
    order = np.random.default_rng(seed).permutation(n)
    result={};offset=0
    for split in ('validation','test'):
        result[split]=order[offset:offset+counts[split]];offset+=counts[split]
    return result


def nearby_seed(metadata,anchor,attempt,split_id,flow_index,spec,components,axes,oyf,train_tree):
    from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar
    near=spec['nearby'];d=float(metadata['neighbor_distance'][anchor])
    rng=np.random.default_rng(np.random.SeedSequence([near['seed'],flow_index,split_id,int(anchor),attempt]))
    direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
    fraction=rng.uniform(*near['offset_neighbor_distance_range'])
    center=metadata['center'][anchor]+direction*(fraction*d)
    closest=float(train_tree.query(center)[0])
    if closest < near['minimum_distance_to_any_train_center_over_anchor_distance']*d:
        return None,'too_close_to_an_original_center'
    offsets=np.array([[x,y,z] for z in (-1,0,1) for y in (-1,0,1) for x in (-1,0,1)],float)
    offsets[[0,13]]=offsets[[13,0]]
    seeds=center+offsets*d
    fluct,inside,cells=interpolate_scalar(seeds,axes,oyf)
    head=int(metadata['head_component'][anchor])
    valid=inside&(fluct>0)&(components.ravel()[cells]==head)
    if not valid[0] or valid.sum()<spec['bundles']['minimum_valid_lines']:
        return None,'insufficient_valid_head_seeds'
    return dict(center=center,seeds=seeds[valid],source_cell=int(cells[0]),neighbor_distance=d,
        seed_rms_distance=float(np.sqrt(np.mean(np.sum((seeds[valid]-center)**2,axis=1)))),
        head_component=head,label=int(metadata['labels'][anchor]),instance=int(metadata['instance'][anchor]),
        center_number=1000000+split_id*100000+int(anchor),scale_id=int(metadata['scale_id'][anchor]),
        anchor_row=int(anchor),anchor_center=metadata['center'][anchor].copy(),
        offset_over_neighbor_distance=float(np.linalg.norm(center-metadata['center'][anchor])/d),
        nearest_train_center_distance=closest),None


def audit_near_metadata(meta,original,expected_anchors,spec,previous_centers=None):
    from scipy.spatial import cKDTree
    anchors=meta['anchor_row'];n=len(anchors)
    if not np.array_equal(anchors,expected_anchors) or len(np.unique(anchors))!=n:
        raise ValueError('Anchor identity or order changed')
    for key in ('labels','head_component','instance','scale_id','neighbor_distance'):
        if not np.array_equal(meta[key],original[key][anchors]):
            raise ValueError('Original anchor field changed: '+key)
    if not np.array_equal(meta['anchor_center'],original['center'][anchors]):
        raise ValueError('Anchor coordinate changed')
    distance=np.linalg.norm(meta['center']-meta['anchor_center'],axis=1)
    relative=distance/meta['neighbor_distance'];lo,hi=spec['nearby']['offset_neighbor_distance_range']
    if np.any(relative<lo-1e-9) or np.any(relative>hi+1e-9):
        raise ValueError('New point is outside prescribed nearby shell')
    nearest=cKDTree(original['center']).query(meta['center'])[0]
    if np.any(nearest<spec['nearby']['minimum_distance_to_any_train_center_over_anchor_distance']*meta['neighbor_distance']):
        raise ValueError('New center coincides with an original center')
    if not np.allclose(relative,meta['offset_over_neighbor_distance'],atol=1e-12,rtol=1e-12):
        raise ValueError('Stored offset differs from coordinates')
    if len(np.unique(meta['center'],axis=0))!=n:
        raise ValueError('Repeated newly generated center')
    if previous_centers is not None:
        all_centers=np.concatenate((previous_centers,meta['center']))
        if len(np.unique(all_centers,axis=0))!=len(all_centers):
            raise ValueError('Nearby validation and test share an exact new center')
    return dict(samples=n,anchor_offset_relative_min_median_max=np.quantile(relative,[0,.5,1]).tolist(),
        nearest_train_center_distance_min_median_max=np.quantile(nearest,[0,.5,1]).tolist(),
        exact_train_center_overlap=0,distinct_anchors=n,
        class_counts=np.bincount(meta['labels'],minlength=2).tolist(),
        shared_training_heads_and_source_support=True)


def prepare(spec,config,index,preflight=False):
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from FMT_Utils.Task4C_PaperBundles_3_1 import load_flow,native_partition
    from FMT_Utils.Task4C_Multiscale_4_1 import extract_native_partition,trace_primitive_batch
    if preflight:
        spec=copy.deepcopy(spec);spec['output']=str(Path(spec['output'])/'preflight')
        spec['nearby']['per_flow_counts']={'validation':8,'test':8}
        spec['sampling']['minimum_class_count']=0
    flow=spec['flows'][index];name=flow['name'];parent=Path(spec['parent_output'])/name
    previous=json.loads((parent/'preparation.json').read_text())
    if previous['identity']['config_sha256']!=spec['parent_config_sha256'] or not previous['spatial_isolation_verified']:
        raise ValueError('Original physical training set changed')
    if previous['flow']!=flow:raise ValueError('Original flow configuration changed')
    for p,h in previous['identity']['source_sha256'].items():
        if pipeline.sha(p)!=h:raise ValueError('Original source changed: '+p)
    for filename in ('geometry.npy','seeds.npy','metadata.npz'):
        if pipeline.sha(parent/'train'/filename)!=previous['splits']['train']['files'][filename]:
            raise ValueError('Original training artifact changed')
    with np.load(parent/'train/metadata.npz') as z:original={k:z[k] for k in z.files}
    if len(original['labels'])!=13500:raise ValueError('Expected all 13500 original training anchors')
    planned=anchor_plan(len(original['labels']),spec['nearby']['per_flow_counts'],spec['nearby']['seed']+index)
    root=Path(spec['input_root'])
    for role in ('flow','gt'):
        if pipeline.sha(root/flow[role])!=flow[role+'_sha256']:raise ValueError('Frozen flow/GT changed')
    axes,_,heads,oyf,grid,source=load_flow(root/flow['flow'],flow['lambda2_threshold'])
    components,count=ndimage.label(heads)
    if count!=previous['catalog']['head_components']:raise ValueError('Native head IDs changed')
    if not np.array_equal(components.ravel()[original['source_cell']],original['head_component']):
        raise ValueError('Anchor head membership changed')
    left,right,support=native_partition(axes,np.asarray(previous['catalog']['x_cuts']),0,
        int(source['vorticity_source'].startswith('second')))
    if support!=previous['splits']['train']['native_source_x_nodes_including_curl_stencil']:
        raise ValueError('Original native training support changed')
    mesh=extract_native_partition(grid,axes,left,right);wall_span=float(axes[2][-1]-axes[2][0])
    tree=cKDTree(original['center']);out=Path(spec['output'])/'physical'/name;out.mkdir(parents=True,exist_ok=False)
    report=dict(version=spec['version'],identity=identity(config),flow=flow,source=source,splits={},
        original_training_preparation_sha256=pipeline.sha(parent/'preparation.json'),
        original_training_metadata_sha256=pipeline.sha(parent/'train/metadata.npz'),
        shared_training_heads_and_source_support=True,spatial_holdout=False,
        native_source_x_nodes_including_curl_stencil=support,preflight=preflight)
    prior_centers=None
    for split_id,split in enumerate(('validation','test'),1):
        anchors=planned[split];attempts={int(a):0 for a in anchors};accepted={};rejections={}
        while len(accepted)<len(anchors):
            pending={s:[] for s in range(len(spec['scale_sets']['train']))}
            for anchor in anchors:
                anchor=int(anchor)
                if anchor in accepted:continue
                attempt=attempts[anchor];attempts[anchor]+=1
                if attempt>=spec['nearby']['maximum_attempts_per_anchor']:
                    raise ValueError('Unable to re-integrate prescribed nearby anchor '+str(anchor))
                sample,reason=nearby_seed(original,anchor,attempt,split_id,index,spec,components,axes,oyf,tree)
                if sample is None:rejections[reason]=rejections.get(reason,0)+1
                else:pending[sample['scale_id']].append(sample)
            for scale_id,samples in pending.items():
                for start in range(0,len(samples),spec['sampling']['trace_batch_primitives']):
                    good,bad,_=trace_primitive_batch(mesh,samples[start:start+spec['sampling']['trace_batch_primitives']],
                        spec['scale_sets']['train'][scale_id],spec,wall_span,(axes[0][left],axes[0][right]))
                    for r in good:accepted[r['anchor_row']]=r
                    for r in bad:rejections[r['reason']]=rejections.get(r['reason'],0)+1
            pipeline.write(out/(split+'_progress.json'),dict(accepted=len(accepted),target=len(anchors),
                total_attempts=sum(attempts.values()),rejections=rejections))
            print(json.dumps(dict(flow=name,split=split,accepted=len(accepted),target=len(anchors))),flush=True)
        rows=[accepted[int(a)] for a in anchors]
        extras={k:np.asarray([r[k] for r in rows]) for k in ('anchor_row','anchor_center',
            'offset_over_neighbor_distance','nearest_train_center_distance')}
        entry=physical.save_split(rows,out/split,spec)
        with np.load(out/split/'metadata.npz') as z:meta={k:z[k] for k in z.files}
        meta.update(extras);np.savez_compressed(out/split/'metadata.npz',**meta)
        entry['files']['metadata.npz']=pipeline.sha(out/split/'metadata.npz')
        entry['nearby_audit']=audit_near_metadata(meta,original,anchors,spec,prior_centers)
        entry['rejections']=rejections;entry['attempts_max']=max(attempts.values())
        report['splits'][split]=entry;prior_centers=meta['center']
    report['completed_at_utc']=datetime.now(timezone.utc).isoformat()
    pipeline.write(out/'preparation.json',report)


def encode(spec,config):
    source=Path(spec['cache_source']['output'])
    if pipeline.sha(source/'encoding.json')!=spec['cache_source']['encoding_sha256']:
        raise ValueError('Original encoded manifest changed')
    original=json.loads((source/'encoding.json').read_text())
    if original['identity']['config_sha256']!=spec['cache_source']['config_sha256']:
        raise ValueError('Original encoding config changed')
    for name,h in original['identity']['source_sha256'].items():
        if pipeline.sha(name)!=h:raise ValueError('Frozen encoder changed: '+name)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.environ.get('SLURM_JOB_ID') and device.type!='cuda':raise ValueError('Encoder GPU unavailable')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    out=Path(spec['output']);report=dict(version=spec['version'],identity=identity(config),splits={},
        test_scope='new_nearby_centers_only_not_original_4.1_test',test_encoded=True)
    for flow in spec['flows']:
        name=flow['name'];key=name+'/train';destination=out/key;destination.mkdir(parents=True,exist_ok=False)
        for filename in ('fmt.npy','voxels.npy','metadata.npz'):
            if pipeline.sha(source/key/filename)!=original['splits'][key]['files'][filename]:raise ValueError('Train cache changed')
            shutil.copyfile(source/key/filename,destination/filename)
            if pipeline.sha(destination/filename)!=original['splits'][key]['files'][filename]:raise ValueError('Copy failed')
        report['splits'][key]=copy.deepcopy(original['splits'][key])
        physical_root=out/'physical'/name;prepared=json.loads((physical_root/'preparation.json').read_text())
        if prepared['identity']['config_sha256']!=pipeline.sha(config):raise ValueError('Nearby preparation config changed')
        for split in ('validation','test'):
            key=name+'/'+split;src=physical_root/split;dst=out/key;dst.mkdir(parents=True,exist_ok=False)
            for filename,h in prepared['splits'][split]['files'].items():
                if pipeline.sha(src/filename)!=h:raise ValueError('Nearby geometry changed')
            with np.load(src/'metadata.npz') as z:meta={k:z[k] for k in z.files}
            n=len(meta['labels']);g=np.load(src/'geometry.npy',mmap_mode='r');seeds=np.load(src/'seeds.npy',mmap_mode='r')
            fmt=np.lib.format.open_memmap(dst/'fmt.npy',mode='w+',dtype='float32',shape=(n,27,234))
            voxels=np.lib.format.open_memmap(dst/'voxels.npy',mode='w+',dtype='float16',shape=(n,4,24,24,24))
            for start in range(0,n,32):
                sl=slice(start,start+32);geometry=torch.as_tensor(np.array(g[sl]),device=device)
                ss=torch.as_tensor(np.array(seeds[sl]),device=device);counts=torch.as_tensor(meta['counts'][sl].astype(np.int64),device=device)
                fmt[sl]=line_fmt(geometry,ss,counts).cpu().numpy()
                voxels[sl]=bundle_voxels(geometry,counts).cpu().numpy().astype(np.float16)
            fmt.flush();voxels.flush();shutil.copyfile(src/'metadata.npz',dst/'metadata.npz')
            report['splits'][key]=dict(samples=n,class_counts=np.bincount(meta['labels'],minlength=2).tolist(),
                files={f:pipeline.sha(dst/f) for f in ('fmt.npy','voxels.npy','metadata.npz')})
    pipeline.write(out/'encoding.json',report)


def plan(spec):
    return [(method,candidate,seed) for candidate in spec['candidates'] for seed in spec['training']['seeds'] for method in spec['methods']]


def run_folder(spec,method,candidate,seed):
    return Path(spec['output'])/'runs'/f"{candidate['name']}_{method}_seed{seed}"


def load_data(spec,split,method,manifest,allow_test=False):
    if split not in ('train','validation') and not (split=='test' and allow_test):
        raise ValueError('Nearby test may only be read after model selection')
    return pipeline.load_data(spec,split,method,manifest,dict(architecture='learned_pool_wider_conv_4.3'))


def train(spec,config,index):
    method,candidate,seed=plan(spec)[index];out=run_folder(spec,method,candidate,seed);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((Path(spec['output'])/'encoding.json').read_text())
    if manifest['identity']['config_sha256']!=pipeline.sha(config):raise ValueError('Encoding config changed')
    options=spec['training'];torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.environ.get('SLURM_JOB_ID') and device.type!='cuda':raise ValueError('Training GPU unavailable')
    started=time.time();training=load_data(spec,'train',method,manifest);validation=load_data(spec,'validation',method,manifest)
    n=len(training['labels']);assert n==spec['expected_training_samples']
    mean=std=None
    def standardize(data,fit=False):
        nonlocal mean,std
        if method!='fmt_mlp':return
        mask=data['features'][...,-1:];x=transform_fmt(data['features'][...,:-1],True)
        if fit:
            valid=x[mask[...,0]>.5];mean=valid.mean(0,dtype=np.float64);std=valid.std(0,dtype=np.float64);std[std<1e-8]=1
        x=np.clip(((x-mean)/std).astype(np.float32),-8.,8.)
        data['features']=np.concatenate((x*mask,mask),-1)
    standardize(training,True);standardize(validation)
    model=make_model(method,dict(candidate,architecture='learned_pool_wider_conv_4.3')).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=options['learning_rate'],weight_decay=candidate['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,
        patience=options['lr_patience'],threshold=1e-4,min_lr=1e-6)
    batch=options['batch_size']
    def predict(data):
        model.eval();p=[];loss=0.
        with torch.no_grad():
            for start in range(0,len(data['labels']),batch):
                x=torch.as_tensor(data['features'][start:start+batch],device=device,dtype=torch.float32)
                y=torch.as_tensor(data['labels'][start:start+batch],device=device,dtype=torch.long)
                logits=model(x);loss+=float(F.cross_entropy(logits,y,reduction='sum'));p.append(logits.softmax(-1)[:,1].cpu().numpy())
        return np.concatenate(p),loss/len(data['labels'])
    best=(-1.,-1.);best_epoch=0;state=None;history=[]
    for epoch in range(1,options['epochs']+1):
        order=rng.permutation(n);model.train();total=0.;lr=optimizer.param_groups[0]['lr']
        for start in range(0,n,batch):
            ids=order[start:start+batch]
            x=torch.as_tensor(training['features'][ids],device=device,dtype=torch.float32)
            y=torch.as_tensor(training['labels'][ids],device=device,dtype=torch.long)
            optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*len(ids)
        probability,val_loss=predict(validation);values=pipeline.binary_metrics(validation['labels'],probability,.5)
        score=(values['f1'],values['average_precision'])
        row=dict(epoch=epoch,training_loss=total/n,validation_loss=val_loss,validation_f1=score[0],
            validation_average_precision=score[1],learning_rate=lr,training_examples=n,
            permutation_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),seconds=time.time()-started)
        history.append(row)
        with (out/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        pipeline.write(out/'progress.json',dict(method=method,candidate=candidate,seed=seed,**row))
        if score>best:best=score;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
        scheduler.step(val_loss)
        if epoch==1 or epoch%10==0:print(json.dumps(dict(method=method,candidate=candidate['name'],seed=seed,**row)),flush=True)
        if epoch-best_epoch>=options['patience']:break
    model.load_state_dict(state);val_probability,_=predict(validation)
    if abs(pipeline.binary_metrics(validation['labels'],val_probability,.5)['f1']-best[0])>1e-12:
        raise ValueError('Validation state restore mismatch')
    threshold=pipeline.choose_threshold(validation['labels'],val_probability)
    lock=dict(identity=identity(config),method=method,candidate=candidate,seed=seed,selected_epoch=best_epoch,
        fixed_threshold=.5,validation_selected_threshold=threshold,test_loaded=False,
        selected_at_utc=datetime.now(timezone.utc).isoformat())
    pipeline.write(out/'selection.lock.json',lock)
    train_probability,_=predict(training)
    test=load_data(spec,'test',method,manifest,allow_test=True);standardize(test);test_probability,_=predict(test)
    probabilities={}
    for split,data,p in (('train',training,train_probability),('validation',validation,val_probability),('test',test,test_probability)):
        for key in ('labels','flow_index','head_component','scale_id'):probabilities[split+'_'+key]=data[key]
        probabilities[split+'_probability']=p
    np.savez_compressed(out/'predictions.npz',**probabilities)
    result=dict(version=spec['version'],identity=identity(config),method=method,candidate=candidate,seed=seed,
        parameters=sum(p.numel() for p in model.parameters()),selected_epoch=best_epoch,threshold=.5,
        training=pipeline.all_metrics(training,train_probability,.5,spec),validation=pipeline.all_metrics(validation,val_probability,.5,spec),
        test=pipeline.all_metrics(test,test_probability,.5,spec),validation_selected_threshold=threshold,
        test_with_validation_selected_threshold=pipeline.all_metrics(test,test_probability,threshold,spec),
        history=history,training_examples=n,validation_examples=len(validation['labels']),test_examples=len(test['labels']),
        selection_lock_sha256=pipeline.sha(out/'selection.lock.json'),predictions_sha256=pipeline.sha(out/'predictions.npz'),
        weights_written=0,normalization=None if mean is None else dict(mean=mean.tolist(),std=std.tolist()),
        test_scope='new_nearby_centers_known_training_heads',original_spatial_test_loaded=False,
        started_at_utc=datetime.fromtimestamp(started,timezone.utc).isoformat(),finished_at_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.time()-started,device=torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',torch=torch.__version__)
    pipeline.write(out/'result.json',result)


def merge(spec,config):
    records=[]
    for method,candidate,seed in plan(spec):
        out=run_folder(spec,method,candidate,seed);r=json.loads((out/'result.json').read_text())
        if r['identity']['config_sha256']!=pipeline.sha(config) or r['predictions_sha256']!=pipeline.sha(out/'predictions.npz'):
            raise ValueError('Run identity changed')
        records.append(r)
    rows=[]
    for method in spec['methods']:
        for candidate in spec['candidates']:
            rr=[r for r in records if r['method']==method and r['candidate']==candidate]
            row=dict(method=method,candidate=candidate,seeds=[r['seed'] for r in rr],parameters=rr[0]['parameters'])
            for split in ('training','validation','test'):
                values=[r[split]['pooled']['f1'] for r in rr]
                row[split]=dict(f1_mean=float(np.mean(values)),f1_std=float(np.std(values,ddof=1)),per_seed_f1=values,
                    per_flow_f1_mean={flow['name']:float(np.mean([r[split]['per_flow'][flow['name']]['f1'] for r in rr])) for flow in spec['flows']})
            rows.append(row)
    summary=dict(version=spec['version'],identity=identity(config),rows=rows,
        comparison='report_both_prespecified_regularization_arms_no_test_selection',
        original_spatial_test_loaded=False,completed_at_utc=datetime.now(timezone.utc).isoformat())
    pipeline.write(Path(spec['output'])/'summary.json',summary);print(json.dumps(summary),flush=True)


def runtime(spec,config,phase,state,code=None):
    row=dict(time=datetime.now(timezone.utc).isoformat(),phase=phase,state=state,exit_code=code,
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',**identity(config))
    pipeline.append_locked(Path(spec['output'])/'runtime_events.jsonl',json.dumps(row)+'\n')
    pipeline.append_locked('docs/ibex_run_registry.md','\n- Task4-c 4.13 runtime '+json.dumps(row)+'\n')


def submit(spec,config):
    out=Path(spec['output']).resolve();out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir()
    pipeline.write(out/'config.frozen.json',spec);previous=None
    for phase,count in (('preflight',2),('prepare',2),('encode',None),('train',len(plan(spec))),('merge',None)):
        gpu=phase in ('encode','train')
        command=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=32G',
            '--time=04:00:00' if phase in ('prepare','train') else '--time=01:00:00',f'--job-name=t4c413-{phase}',
            f'--output={out}/logs/{phase}_%A_%a.out',f'--error={out}/logs/{phase}_%A_%a.err']
        if previous:command.extend((f'--dependency=afterok:{previous}','--kill-on-invalid-dep=yes'))
        if gpu:command.extend(('--gres=gpu:1','--constraint=a100|v100'))
        if count is not None:command.append(f'--array=0-{count-1}%4')
        command.extend(('ibex_bash/task4c_nearby_sampling_4p13.sh',phase,config))
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row=dict(job_id=job,phase=phase,version=spec['version'],submitted_at_utc=datetime.now(timezone.utc).isoformat(),
            command=command,dependency=previous,expected_device='A100/V100' if gpu else 'CPU',**identity(config))
        pipeline.append_locked(out/'submissions.jsonl',json.dumps(row)+'\n')
        pipeline.append_locked('docs/ibex_run_registry.md','\n- Task4-c 4.13 SUBMITTED '+json.dumps(row)+'\n')
        print(json.dumps(row),flush=True);previous=job


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('preflight','prepare','encode','train','merge','submit','runtime'))
    p.add_argument('--config',default=DEFAULT_CONFIG);p.add_argument('--index',type=int,default=0)
    p.add_argument('--runtime-phase');p.add_argument('--state');p.add_argument('--exit-code',type=int)
    a=p.parse_args();spec=load_spec(a.config)
    if a.phase in ('prepare','preflight'):prepare(spec,a.config,a.index,a.phase=='preflight')
    elif a.phase=='train':train(spec,a.config,a.index)
    elif a.phase=='runtime':runtime(spec,a.config,a.runtime_phase,a.state,a.exit_code)
    else:{'encode':encode,'merge':merge,'submit':submit}[a.phase](spec,a.config)


if __name__=='__main__':main()
