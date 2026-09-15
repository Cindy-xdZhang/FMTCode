"""Twenty preregistered P35 normalization/frequency combinations, validation only."""
from __future__ import annotations

import argparse
import collections
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import FMTv8_Search_2_2 as legacy
from experiments.FMTv8_Search_2_2 import (old, old4, write, identity, parents, units,
    execution_settings, cache_folder, save_cache, read_cache, train_backbones,
    task4_model, residual_spec, pad_auxiliary, PROFILES, append_record)
from FMT_Utils.FMT_P35_NormFrequency_3_1 import (candidates, encode, primitive_features,
    normalize_geometry, fit_normalizer, apply_normalizer, FREQUENCIES)

DEFAULT_CONFIG='config/Ablation_FMTv8_NormFrequency_3.1.json'


def load_spec(config):
    spec=json.loads(Path(config).read_text())
    for name,expected in spec['frozen_config_hashes'].items():assert old.sha(name)==expected,name
    assert spec['candidates']==candidates()
    assert len(spec['candidates'])==20
    return spec


def cleanup_weights(spec,directory):
    root=Path(spec['output']).resolve();directory=Path(directory).resolve()
    rel=directory.relative_to(root)
    if not rel.parts or rel.parts[0] not in ('engineering_smoke','cache','controls','search'):
        raise ValueError('Unexpected cleanup directory: '+str(directory))
    for extension in ('*.pt','*.pth','*.ckpt'):
        for path in directory.rglob(extension):
            resolved=path.resolve();resolved.relative_to(directory)
            if path.is_symlink():raise ValueError('Linked checkpoint')
            row=dict(path=str(resolved),sha256=old.sha(resolved),time=time.time())
            resolved.unlink()
            append_record(root/'weight_cleanup.jsonl',row,'FMTv8 NormFrequency 3.1 temporary weight removed')


def read_source(spec,task,dataset,role):
    if role not in ('train','validation'):raise ValueError('This experiment never reads test data')
    data,evidence=legacy.read_source(spec,task,dataset,role)
    if task!='Task4C':return data,evidence
    _,b4=parents(spec);root=Path(b4['parent_output'])/'physical';parts=[]
    for flow in b4['flows']:
        folder=root/flow['name'];manifest=folder/'preparation.json'
        assert old.sha(manifest)==spec['task4_preparation_hashes'][flow['name']]
        record=json.loads(manifest.read_text());items={}
        for name in ('geometry.npy','seeds.npy','metadata.npz'):
            path=folder/role/name;expected=record['splits'][role]['files'][name]
            assert old.sha(path)==expected
            evidence.append(dict(path=str(path),sha256=expected))
        items['geometry']=np.load(folder/role/'geometry.npy')
        items['seeds']=np.load(folder/role/'seeds.npy')
        with np.load(folder/role/'metadata.npz') as z:
            items.update(counts=z['counts'],labels=z['labels'])
        parts.append(items)
    assert np.array_equal(data['labels'],np.concatenate([d['labels'] for d in parts]))
    for key in ('geometry','seeds','counts'):data[key]=np.concatenate([d[key] for d in parts])
    expected=np.arange(data['geometry'].shape[1])[None]<data['counts'][:,None]
    assert np.array_equal(expected,data['mask'][...,0]>.5)
    return data,evidence


def prepare(spec,config,index):
    task,dataset=units(spec)[index];folder=cache_folder(spec,task,dataset)
    folder.mkdir(parents=True,exist_ok=False);files={};evidence={};data={}
    for role in ('train','validation'):
        data[role],evidence[role]=read_source(spec,task,dataset,role)
        files[role]=save_cache(folder/role,data[role])
    if task=='Task4C':assert len(data['train']['labels'])==27000 and len(data['validation']['labels'])==3000
    else:train_backbones(spec,task,dataset,spec['screen_seed'],data['train'],data['validation'],folder/'temporary_backbones')
    write(folder/'manifest.json',dict(status='PASS',files=files,evidence=evidence,identity=identity(config),test_read=False))
    print('PREPARED',task,dataset,flush=True)


def feature_arrays(data,definition,task):
    result=[];batch=32 if task=='Task4C' else 1024
    geometry=data['geometry'] if task=='Task4C' else data['raw']
    for start in range(0,len(geometry),batch):
        sl=slice(start,start+batch)
        if task=='Task4C':
            # The original 4.14 encoder selected neighbors on V100 in batches of 32.
            # CPU distance rounding can change ties and therefore the six input lines.
            g=torch.as_tensor(np.array(geometry[sl]),device='cuda')
            extra=dict(seeds=torch.as_tensor(np.array(data['seeds'][sl]),device='cuda'),
                       counts=torch.as_tensor(np.array(data['counts'][sl]),device='cuda'))
        else:g=np.array(geometry[sl]);extra={}
        result.append(encode(g,definition,**extra).cpu().numpy())
    return np.concatenate(result)


def residual_fit(spec,config,task,dataset,seed,pool,profile,data,folder,backbones):
    base,_=parents(spec);tr,va=data['train'],data['validation'];width=433
    feature_start=time.perf_counter()
    auxiliary=[feature_arrays(d,pool,task) for d in (tr,va)]
    normalization=fit_normalizer(auxiliary[0],None,pool['features'])
    aux=[pad_auxiliary(apply_normalizer(x,normalization),width) for x in auxiliary]
    nt,nv,_,stats=old._normalize_train_only(
        (tr['raw'],np.zeros_like(aux[0]),tr['labels']),
        (va['raw'],np.zeros_like(aux[1]),va['labels']),None)
    nt=(nt[0],aux[0],nt[2]);nv=(nv[0],aux[1],nv[2])
    stats['p35_normalization']=normalization
    feature_seconds=time.perf_counter()-feature_start
    settings=residual_spec(base,task,profile,backbones/'checkpoints',seed);settings['experiment']=spec['version']
    folder.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    outcome=old.train_residual(settings,dataset,seed,(nt,nv,None),stats,torch.device('cuda'),folder)
    elapsed=time.perf_counter()-started
    model,state=old.load_residual(outcome['checkpoint'],width,torch.device('cuda'))
    y,r,a=old._predict_components(model,old._loader(nv,512,False,seed,True),torch.device('cuda'))
    p=old._probabilities(r,a,state['alpha'],state['config']['model']);threshold=float(state['threshold'])
    score=old.metrics(y,p,threshold)
    assert abs(score['f1']-outcome['validation_f1'])<1e-12
    np.savez_compressed(folder/'validation_predictions.npz',labels=y,probability=p,threshold=threshold)
    backbone_record=json.loads((backbones/'backbones.json').read_text())
    result=dict(task=task,dataset=dataset,seed=seed,pool=pool['id'],profile=profile,
        feature_dimensions=pool['feature_dimensions'],definition=pool,normalization=normalization,feature_seconds=feature_seconds,parameters=state['total_parameter_count'],
        trainable_parameters=state['trainable_residual_parameter_count'],validation=score,
        threshold=threshold,alpha=float(state['alpha']),selected_epoch=state['best_epoch'],
        seconds=elapsed,raw_training_seconds=backbone_record['results']['raw']['seconds'],
        raw_wide_selection_seconds=backbone_record['results']['raw_wide']['seconds'],
        raw_checkpoint_sha256=old.sha(state['raw_checkpoint']),training_result=outcome,test_read=False,
        predictions_sha256=old.sha(folder/'validation_predictions.npz'),identity=identity(config))
    write(folder/'result.json',result)
    return result,(model,state)


def task4_fit(spec,config,seed,pool,profile,data,folder,epochs_override=None):
    _,base=parents(spec);options=copy.deepcopy(base['training']);settings=PROFILES[profile]['task4']
    options.update({key:value for key,value in settings.items() if key in options})
    if epochs_override:options['epochs']=epochs_override
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    folder.mkdir(parents=True,exist_ok=False);started=time.time()
    train,val=data['train'],data['validation'];arrays={};normalization=None
    for role in ('train','validation'):
        d=data[role];x=feature_arrays(d,pool,'Task4C');mask=np.asarray(d['mask'])
        if role=='train':normalization=fit_normalizer(x,mask,pool['features'])
        arrays[role]=np.concatenate((apply_normalizer(x,normalization,mask),mask),-1)
    feature_seconds=time.time()-started
    model=task4_model(pool['feature_dimensions'],profile).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=options['learning_rate'],weight_decay=settings.get('weight_decay',.0001))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,
        patience=options['lr_patience'],threshold=1e-4,min_lr=1e-6)
    batch=options['batch_size'];n=len(train['labels'])
    def predict(x,labels):
        model.eval();probs=[];loss=0.
        with torch.no_grad():
            for start in range(0,len(labels),batch):
                xx=torch.as_tensor(x[start:start+batch],device='cuda');yy=torch.as_tensor(labels[start:start+batch],device='cuda',dtype=torch.long)
                logits=model(xx);probs.append(logits.softmax(-1)[:,1].cpu().numpy());loss+=float(F.cross_entropy(logits,yy,reduction='sum'))
        return np.concatenate(probs),loss/len(labels)
    best=(-1.,-1.);best_epoch=0;state=None;history=[]
    for epoch in range(1,options['epochs']+1):
        order=rng.permutation(n);model.train();loss_sum=0.;lr=optimizer.param_groups[0]['lr']
        for start in range(0,n,batch):
            ids=order[start:start+batch];x=torch.as_tensor(arrays['train'][ids],device='cuda');y=torch.as_tensor(train['labels'][ids],device='cuda',dtype=torch.long)
            optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x),y)
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            loss_sum+=float(loss.detach())*len(ids)
        probability,val_loss=predict(arrays['validation'],val['labels']);metric=old4.pipeline.binary_metrics(val['labels'],probability,.5)
        score=(metric['f1'],metric['average_precision'])
        row=dict(epoch=epoch,training_loss=loss_sum/n,validation_loss=val_loss,validation_f1=score[0],
            validation_average_precision=score[1],learning_rate=lr,training_examples=n,
            permutation_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),seconds=time.time()-started)
        history.append(row)
        with (folder/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if score>best:best=score;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
        scheduler.step(val_loss)
        if epoch==1 or epoch%25==0:print(pool['id'],profile,seed,row,flush=True)
        if epoch-best_epoch>=options['patience']:break
    model.load_state_dict(state);probability,_=predict(arrays['validation'],val['labels'])
    score=old4.pipeline.binary_metrics(val['labels'],probability,.5);assert abs(score['f1']-best[0])<1e-12
    np.savez_compressed(folder/'validation_predictions.npz',labels=val['labels'],probability=probability,threshold=.5,
        **{key:val[key] for key in ('flow_index','head_component','scale_id','instance')})
    result=dict(task='Task4C',dataset='channel_tbl',seed=seed,pool=pool['id'],profile=profile,
        feature_dimensions=pool['feature_dimensions'],parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters()),validation=score,
        threshold=.5,selected_epoch=best_epoch,epochs=len(history),seconds=history[-1]['seconds'],
        raw_training_seconds=0.,raw_wide_selection_seconds=0.,test_read=False,
        normalization=normalization,definition=pool,feature_seconds=feature_seconds,history=history,
        predictions_sha256=old.sha(folder/'validation_predictions.npz'),identity=identity(config))
    write(folder/'result.json',result)
    return result,(model,normalization)


def folder_for(spec,phase,task,dataset,candidate):
    return Path(spec['output'])/phase/task/dataset/candidate


def audit_validation(folder):
    row=json.loads((folder/'result.json').read_text())
    assert row['test_read'] is False and old.sha(folder/'validation_predictions.npz')==row['predictions_sha256']
    with np.load(folder/'validation_predictions.npz') as z:
        y=z['labels'];p=z['probability'];pred=p>=float(z['threshold'])
        tp=int(np.sum((y==1)&pred));fp=int(np.sum((y==0)&pred));fn=int(np.sum((y==1)&~pred))
        f1=2*tp/max(2*tp+fp+fn,1)
        assert abs(f1-row['validation']['f1'])<1e-12 and np.isfinite(p).all()
    return row


def controls(spec,config,index):
    task,dataset=units(spec)[index];seed=spec['screen_task4_seed'] if task=='Task4C' else spec['screen_seed']
    data={role:read_cache(spec,task,dataset,role) for role in ('train','validation')}
    pool=next(p for p in legacy.pooling_candidates() if p['id']=='p35')
    folder=folder_for(spec,'controls',task,dataset,'p35_h0')
    if task=='Task4C':row,item=legacy.task4_fit(spec,config,seed,pool,'h0',data,folder)
    else:row,item=legacy.residual_fit(spec,config,task,dataset,seed,pool,'h0',data,folder,cache_folder(spec,task,dataset)/'temporary_backbones')
    del item;cleanup_weights(spec,folder);audit_validation(folder)
    previous=Path(spec['p35_reference'])/'search'/task/dataset/'p35_h0'/f'seed{seed}'
    before=json.loads((previous/'result.json').read_text())
    assert old.sha(previous/'validation_predictions.npz')==before['predictions_sha256']
    with np.load(previous/'validation_predictions.npz') as a,np.load(folder/'validation_predictions.npz') as b:
        assert set(a.files)==set(b.files)
        for key in a.files:assert np.array_equal(a[key],b[key]),(task,dataset,key)
    assert before['selected_epoch']==row['selected_epoch']
    write(Path(spec['output'])/'control_checks'/f'{task}_{dataset}.json',dict(status='PASS',task=task,dataset=dataset,
        historical_predictions_sha256=before['predictions_sha256'],new_predictions_sha256=row['predictions_sha256'],
        exact_probabilities=True,selected_epoch=row['selected_epoch'],validation_f1=row['validation']['f1'],test_read=False))


def search(spec,config,index):
    checks=list((Path(spec['output'])/'control_checks').glob('*.json'))
    assert len(checks)==21 and all(json.loads(p.read_text())['status']=='PASS' for p in checks)
    task,dataset=units(spec)[index//4];group=index%4
    seed=spec['screen_task4_seed'] if task=='Task4C' else spec['screen_seed']
    data={role:read_cache(spec,task,dataset,role) for role in ('train','validation')}
    for definition in spec['candidates'][group*5:group*5+5]:
        folder=folder_for(spec,'search',task,dataset,definition['id'])
        if task=='Task4C':row,item=task4_fit(spec,config,seed,definition,'h0',data,folder)
        else:row,item=residual_fit(spec,config,task,dataset,seed,definition,'h0',data,folder,cache_folder(spec,task,dataset)/'temporary_backbones')
        del item;cleanup_weights(spec,folder);audit_validation(folder)
        print('FINISHED',task,dataset,definition['id'],row['validation']['f1'],flush=True)


def merge(spec,config):
    rows=[];ranking=[]
    for definition in spec['candidates']:
        items=[]
        for task,dataset in units(spec):
            folder=folder_for(spec,'search',task,dataset,definition['id']);row=audit_validation(folder)
            # Labels are independently checked against the source-verified preparation cache.
            cache=cache_folder(spec,task,dataset);manifest=json.loads((cache/'manifest.json').read_text())
            path=cache/'validation'/'labels.npy';assert old.sha(path)==manifest['files']['validation']['labels.npy']
            with np.load(folder/'validation_predictions.npz') as z:assert np.array_equal(z['labels'],np.load(path))
            items.append(row);rows.append(row)
        scores={task:float(np.mean([r['validation']['f1'] for r in items if r['task']==task])) for task in ('Task3','Task5','Task4C')}
        ranking.append(dict(definition,tasks=scores,score=float(np.mean(list(scores.values()))),
            task4_parameters=next(r['parameters'] for r in items if r['task']=='Task4C'),
            total_fit_seconds=sum(r['seconds'] for r in items)))
    ranking.sort(key=lambda r:(-r['score'],r['feature_dimensions'],r['id']))
    controls=[audit_validation(folder_for(spec,'controls',t,d,'p35_h0')) for t,d in units(spec)]
    baseline={t:float(np.mean([r['validation']['f1'] for r in controls if r['task']==t])) for t in ('Task3','Task5','Task4C')}
    baseline['score']=float(np.mean(list(baseline.values())))
    assert len(rows)==420
    for task,dataset in units(spec):cleanup_weights(spec,cache_folder(spec,task,dataset))
    assert not any(Path(spec['output']).rglob('*.pt'))
    report=dict(status='PASS',version=spec['version'],ranking=ranking,baseline=baseline,rows=rows,
        test_read=False,identity=identity(config),prediction_count=len(rows)+len(controls),
        selected=ranking[0],selection='Equal mean of Task3 ten-flow F1, Task5 ten-flow F1, Task4C pooled F1')
    write(Path(spec['output'])/'summary.json',report)
    table=['# P35 common normalization / Fourier frequency search 3.1','',
           'All scores are the original validation sets used for model and configuration selection. Task3/5 seed40; Task4C seed96611.',
           '', '|Rank|Candidate|Geometry|Feature transform|K|Dimensions|Task3 F1|Task5 F1|Task4C F1|Mean F1|',
           '|---:|---|---|---|---:|---:|---:|---:|---:|---:|']
    for i,r in enumerate(ranking,1):
        table.append(f"|{i}|{r['id']}|{r['geometry']}|{r['features']}|{r['frequencies']}|{r['feature_dimensions']}|{r['tasks']['Task3']:.6f}|{r['tasks']['Task5']:.6f}|{r['tasks']['Task4C']:.6f}|{r['score']:.6f}|")
    table.extend(['', 'Frozen original P35/h0 (reproduced exactly): '+json.dumps(baseline), ''])
    (Path(spec['output'])/'validation_rankings.md').write_text('\n'.join(table))


def numpy_reference(geometry,definition):
    """Independent NumPy geometry/DFT/Gram/pooling reference, no production encoder."""
    x=np.asarray(geometry,dtype=np.float64)
    x=x-x.mean((1,2),keepdims=True)
    radius2=(x*x).sum(-1)
    radius=np.sqrt(radius2.max((1,2)) if definition['geometry']=='max_radius' else radius2.mean((1,2)))
    x=(x/radius[:,None,None,None]).astype(np.float32)
    k=definition['frequencies']
    def descriptor(v):
        spectrum=np.fft.rfft(v.astype(np.float64),axis=1)[:,:k]
        a,b=spectrum.real,spectrum.imag;an=np.linalg.norm(a,axis=-1);bn=np.linalg.norm(b,axis=-1)
        gram=np.stack((an,bn,(a*b).sum(-1)/np.maximum(an*bn,1e-8)),-1).reshape(len(v),-1)
        triple=(np.cross(a[:,:-1],b[:,:-1])*a[:,1:]).sum(-1)
        triple/=np.maximum(an[:,:-1]*bn[:,:-1]*an[:,1:],1e-8)
        return np.concatenate((gram,triple),-1)
    c=descriptor(np.diff(x[:,0],axis=1));delta=np.diff(x[:,1:]-x[:,:1],axis=2)
    n=descriptor(delta.reshape(-1,31,3)).reshape(len(x),6,-1)
    n=np.sort(n,axis=1)[:,::-1]
    tangent=np.diff(x[:,0],axis=1);tangent/=np.maximum(np.linalg.norm(tangent,axis=-1,keepdims=True),1e-12)
    tangent=np.concatenate((tangent,tangent[:,-1:]),1)
    spectrum=np.fft.rfft(np.concatenate((x[:,0],tangent),-1).astype(np.float64),axis=1,norm='ortho')[:,:k]
    signed=np.stack((spectrum.real,spectrum.imag),-1).reshape(len(x),-1)
    return np.concatenate((c,signed,n.mean(1),n.max(1)),-1)


def preflight(spec,config):
    assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0)
    identity(config);torch.manual_seed(2131);checks=[]
    rng=np.random.default_rng(2131);x=rng.normal(size=(11,7,32,3)).astype(np.float32)
    for definition in spec['candidates']:
        features=encode(x,definition).numpy();reference=numpy_reference(x,definition)
        np.testing.assert_allclose(features,reference,rtol=2e-4,atol=2e-5)
        np.testing.assert_allclose(features,encode(x[:,[0,4,2,6,1,3,5]],definition).numpy(),rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(features,encode(x*3+np.array([1,-3,2],np.float32),definition).numpy(),rtol=2e-4,atol=2e-5)
        k=definition['frequencies'];d=4*k-1
        assert features.shape==(11,24*k-3)
        # Imaginary DC norm, DC cosine, first chirality, and signed DC imaginary slots are zero.
        assert np.all(features[:,[1,2,3*k]]==0)
        assert np.all(features[:,d:d+12*k][:,1:12:2]==0)
        norm=fit_normalizer(features,None,definition['features'])
        normal=apply_normalizer(features,norm)
        np.testing.assert_allclose(normal.mean(0),0,atol=2e-6)
        counts=np.array([10,11,12]);g=rng.normal(size=(3,27,32,3)).astype(np.float32)
        seeds=rng.normal(size=(3,27,3)).astype(np.float32);mask=(np.arange(27)[None]<counts[:,None])[...,None]
        g*=mask[...,None];tokens=encode(g,definition,seeds,counts).numpy()
        assert np.all(tokens[~mask[...,0]]==0)
        ns=fit_normalizer(tokens,mask,definition['features'])
        alt=tokens.copy();alt[~mask[...,0]]=100000
        ns2=fit_normalizer(alt,mask,definition['features'])
        assert np.array_equal(ns['mean'],ns2['mean']) and np.array_equal(ns['std'],ns2['std'])
        model=task4_model(definition['feature_dimensions'],'h0').cuda()
        assert sum(p.numel() for p in model.parameters())==58690+128*definition['feature_dimensions']
        tensor=torch.from_numpy(np.concatenate((apply_normalizer(tokens,ns,mask),mask.astype(np.float32)),-1)).cuda()
        loss=model(tensor).square().mean();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        checks.append(dict(candidate=definition['id'],dimensions=features.shape[-1],numpy_reference=True,mask_verified=True))
    # The unchanged K=6 feature composition agrees exactly with the original P35.
    from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
    from FMT_Utils.fmt_v8 import coordinate_tangent_spectrum
    primitive=torch.from_numpy(x)
    base=pathline_dft_features_3d(primitive,num_freq=6,neighbor_scale=1.,neighbor_weight=1.,return_numpy=False)
    source=torch.cat((base,coordinate_tangent_spectrum(primitive[:,0])),-1)
    pool=next(p for p in legacy.pooling_candidates() if p['id']=='p35')
    assert torch.equal(primitive_features(primitive,6),legacy.select_features(source,pool))
    # One-epoch real training fixtures exercise the modified normalizers and frozen trainers.
    # All fixture rows come from training; these scores are engineering checks, never selection inputs.
    from unittest.mock import patch
    base,b4=parents(spec);base=copy.deepcopy(base);b4=copy.deepcopy(b4)
    base['training'].update(max_epochs=1,patience=1);b4['training'].update(epochs=1,patience=1)
    fixtures={};source_device_check=None
    for task,dataset in (('Task3',base['datasets'][0]),('Task4C','channel_tbl')):
        data,_=read_source(spec,task,dataset,'train');ids={c:np.flatnonzero(data['labels']==c) for c in (0,1)}
        if task=='Task4C':
            from FMT_Utils.Task4C_LinePooling_4_3 import line_fmt
            parts=[];changed_sets=0
            for flow in (0,1):
                selected=np.flatnonzero(data['flow_index']==flow)[:32]
                g=torch.as_tensor(data['geometry'][selected],device='cuda')
                s=torch.as_tensor(data['seeds'][selected],device='cuda')
                c=torch.as_tensor(data['counts'][selected].astype(np.int64),device='cuda')
                frozen=line_fmt(g,s,c).cpu().numpy()
                expected=np.concatenate((data['source'][selected],data['mask'][selected]),-1)
                assert np.array_equal(frozen,expected),('Original V100 feature reproduction failed',float(np.max(np.abs(frozen-expected))))
                neighbor_sets=[]
                for device in ('cpu','cuda'):
                    seeds=s.to(device);mask=torch.arange(27,device=device)[None]<c.to(device)[:,None]
                    distance=torch.cdist(seeds,seeds);distance.masked_fill_(~mask[:,None,:],torch.inf)
                    distance.diagonal(dim1=1,dim2=2).fill_(torch.inf)
                    neighbor_sets.append(torch.argsort(distance,dim=-1,stable=True)[...,:6].sort(-1).values.cpu().numpy())
                changed_sets+=int(((neighbor_sets[0]!=neighbor_sets[1]).any(-1)&(data['mask'][selected,...,0]>.5)).sum())
                parts.append(dict(flow_index=flow,samples=32,frozen_v100_features_exact=True))
            source_device_check=dict(parts=parts,cpu_changed_neighbor_sets=changed_sets,batch_size=32,selected_device='V100')
        n=64 if task=='Task3' else 32
        slices={'train':np.r_[ids[0][:n],ids[1][:n]],'validation':np.r_[ids[0][n:n+16],ids[1][n:n+16]]}
        fixtures[task]={role:{key:values[ix] for key,values in data.items()} for role,ix in slices.items()}
    smoke=Path(spec['output'])/'engineering_smoke'/('attempt'+str(spec.get('execution_revision',1)))
    with patch(__name__+'.parents',return_value=(base,b4)),patch('experiments.FMTv8_Search_2_2.parents',return_value=(base,b4)):
        backbones=smoke/'temporary_backbones'
        train_backbones(spec,'Task3',base['datasets'][0],0,fixtures['Task3']['train'],fixtures['Task3']['validation'],backbones)
        for definition in (spec['candidates'][0],spec['candidates'][9],spec['candidates'][10],spec['candidates'][19]):
            folder=smoke/'Task3'/definition['id']
            _,item=residual_fit(spec,config,'Task3',base['datasets'][0],0,definition,'h0',fixtures['Task3'],folder,backbones)
            del item;audit_validation(folder)
            folder=smoke/'Task4C'/definition['id']
            task4_fit(spec,config,0,definition,'h0',fixtures['Task4C'],folder);audit_validation(folder)
    cleanup_weights(spec,smoke)
    write(Path(spec['output'])/f"preflight_r{spec.get('execution_revision',1)}.json",dict(status='PASS',identity=identity(config),checks=checks,
        frozen_p35_composition_exact=True,real_training_only_smoke=True,source_device_check=source_device_check,test_read=False))


def runtime(spec,config,phase,state,code=None):
    row=dict(version=spec['version'],phase=phase,state=state,exit_code=code,time=datetime.now(timezone.utc).isoformat(),
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',**identity(config))
    root=Path(spec['output']);write(root/'lifecycle_events'/f"{row['job']}_{state}_{time.time_ns()}.json",row)
    append_record(root/'runtime_events.jsonl',row,'FMTv8 NormFrequency 3.1 runtime')
    append_record('docs/ibex_run_registry.md',row,'FMTv8 NormFrequency 3.1 runtime')


def submit(spec,config,phase,dependency=None):
    logs=Path(spec['output'])/'logs';logs.mkdir(parents=True,exist_ok=True)
    gpu=phase!='merge';count={'prepare':21,'controls':21,'search':84}.get(phase)
    cmd=['sbatch','--parsable','--cpus-per-task=4','--mem=48G','--time=04:00:00','--job-name=v8nf31-'+phase,
         '--output='+str(logs/(phase+'.%A_%a.out')),'--error='+str(logs/(phase+'.%A_%a.err'))]
    if gpu:cmd+=['--gres=gpu:1','--constraint=v100']
    if count:cmd+=['--array=0-'+str(count-1)+'%'+str(spec['maximum_concurrent_gpus'])]
    if dependency:cmd+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
    cmd+=['ibex_bash/fmtv8_norm_frequency_3p1.sh',phase,config]
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    row=dict(version=spec['version'],phase=phase,job_id=job,command=cmd,submitted_at_utc=datetime.now(timezone.utc).isoformat(),
        config=config,expected_device='V100' if gpu else 'CPU',**identity(config))
    append_record(Path(spec['output'])/'submissions.jsonl',row,'FMTv8 NormFrequency 3.1 submitted')
    append_record('docs/ibex_run_registry.md',row,'FMTv8 NormFrequency 3.1 submitted')
    print(json.dumps(row),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','prepare','controls','search','merge','runtime','submit'])
    p.add_argument('--config',default=DEFAULT_CONFIG);p.add_argument('--index',type=int,default=0)
    p.add_argument('--submit-phase');p.add_argument('--dependency');p.add_argument('--runtime-phase');p.add_argument('--state');p.add_argument('--exit-code',type=int)
    a=p.parse_args();s=load_spec(a.config);execution_settings();torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    if a.phase=='runtime':runtime(s,a.config,a.runtime_phase,a.state,a.exit_code)
    elif a.phase=='submit':submit(s,a.config,a.submit_phase,a.dependency)
    elif a.phase in ('prepare','controls','search'):globals()[a.phase](s,a.config,a.index)
    else:globals()[a.phase](s,a.config)


if __name__=='__main__':main()
