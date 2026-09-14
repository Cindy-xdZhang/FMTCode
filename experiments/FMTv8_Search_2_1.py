"""Validation-only 50-pooling search; fixed shortlist and paired final tests."""
from __future__ import annotations

import argparse
import collections
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from FMT_Utils.FMT_V8_Search_2_1 import (pooling_candidates, PROFILES, pool_neighbors,
    pool_numpy, select_features, pathline_source, task4_model, residual_spec)
from FMT_Utils.fmt_v8 import compose_features
from FMT_Utils.FMT_V8_Comparison_3D import shared_features
from FMT_Utils.Task4C_Multiscale_4_1 import transform_fmt
from FMT_Utils.GeometricControls_3D import pad_auxiliary
from FMT_Utils.PathlineClassifier_3D import PathlineBinaryClassifier3D, PathlineFMTResidualClassifier3D, residual_model_kwargs
from experiments import Run_Task1235_AngleFeatures_1_1 as old
from experiments.Verify_Task3_FMTClassifier import _set_seed
from experiments import Task4C_NearbySampling_4_13 as old4
from experiments.Record_Task1235_AngleFeatures_1_1 import append_record

DEFAULT_CONFIG='config/Ablation_FMTv8_Search_2.1.json'


def write(path, value):
    old.write_json(Path(path),value)


def cleanup_weights(spec,directory):
    """Delete only this experiment's known temporary model files, with receipts."""
    root=Path(spec['output']).resolve();directory=Path(directory).resolve()
    relative=directory.relative_to(root)
    if not relative.parts or relative.parts[0] not in ('engineering_smoke','cache','final_backbones','search','refine','final'):
        raise ValueError('Unexpected cleanup directory: '+str(directory))
    deleted=[]
    for extension in ('*.pt','*.pth','*.ckpt'):
        for path in directory.rglob(extension):
            resolved=path.resolve();resolved.relative_to(directory)
            if path.is_symlink():raise ValueError('Refusing to delete linked checkpoint')
            row=dict(path=str(resolved),sha256=old.sha(resolved),time=time.time())
            resolved.unlink();deleted.append(row)
            append_record(root/'weight_cleanup.jsonl',row,'FMTv8 Search 2.1 temporary weight removed')
    return deleted


def identity(config):
    manifest=json.loads(Path('SOURCE_MANIFEST.json').read_text())
    for name,expected in manifest.items():
        assert old.sha(name)==expected,name
    return dict(commit=Path('SOURCE_COMMIT.txt').read_text().strip(),config_sha256=old.sha(config),
        source_manifest_sha256=old.sha('SOURCE_MANIFEST.sha256'),job=os.environ.get('SLURM_JOB_ID'),
        array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),host=socket.gethostname())


def load_spec(config):
    spec=json.loads(Path(config).read_text())
    for name,expected in spec['frozen_config_hashes'].items():assert old.sha(name)==expected,name
    assert spec['pools']==pooling_candidates() and spec['profiles']==PROFILES
    return spec


def parents(spec):
    base=json.loads(Path(spec['task135_config']).read_text())
    base4=json.loads(Path(spec['task4_config']).read_text())
    physical=json.loads(Path(base4['parent_config']).read_text())
    base4['flows']=json.loads(Path(physical['base_config']).read_text())['flows']
    return base,base4


def units(spec):
    base,_=parents(spec)
    return [(t,d) for t in ('Task3','Task5') for d in base['datasets']]+[('Task4C','channel_tbl')]


def read_source(spec,task,dataset,role,allow_test=False):
    if role not in ('train','validation') and not allow_test:raise ValueError('Search cannot read test')
    base,b4=parents(spec)
    if task=='Task4C':
        root=Path(b4['parent_output']);manifest=root/'encoding.json'
        assert old.sha(manifest)==spec['task4_encoding_sha256']
        manifest=json.loads(manifest.read_text());parts=[];evidence=[]
        for fi,flow in enumerate(b4['flows']):
            key=flow['name']+'/'+role;folder=root/key
            for name in ('fmt.npy','metadata.npz'):
                assert old.sha(folder/name)==manifest['splits'][key]['files'][name]
                evidence.append(dict(path=str(folder/name),sha256=old.sha(folder/name)))
            x=np.load(folder/'fmt.npy');assert x.shape[-1]==234
            with np.load(folder/'metadata.npz') as z:
                data={key:np.asarray(z[key]) for key in ('labels','scale_id','head_component','instance')}
            data.update(source=x[...,:233],mask=x[...,-1:],flow_index=np.full(len(x),fi,np.int8))
            assert not np.any(data['source'][data['mask'][...,0]==0])
            parts.append(data)
        return {key:np.concatenate([d[key] for d in parts]) for key in parts[0]},evidence
    pre=Path(spec['task135_reference'])/'preflight.json'
    assert old.sha(pre)==spec['task135_preflight_sha256']
    rows=old.records(base,task,dataset,'confirmation' if role=='test' else role)
    old.verify_sources(json.loads(pre.read_text()),task,dataset,rows)
    raw=old.raw_array(rows);cached=np.concatenate([r['cached_fmt'] for r in rows])
    data=dict(raw=raw,source=pathline_source(raw,cached),labels=old.labels(rows))
    if task=='Task5':data['scale_id']=np.concatenate([r['scale_id'] for r in rows])
    return data,old.evidence(rows)


def cache_folder(spec,task,dataset):return Path(spec['output'])/'cache'/task/dataset


def save_cache(folder,data):
    folder.mkdir(parents=True,exist_ok=False)
    for name,value in data.items():np.save(folder/(name+'.npy'),value)
    return {p.name:old.sha(p) for p in folder.glob('*.npy')}


def read_cache(spec,task,dataset,role):
    assert role in ('train','validation')
    folder=cache_folder(spec,task,dataset);manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['status']=='PASS'
    result={}
    for name,expected in manifest['files'][role].items():
        p=folder/role/name;assert old.sha(p)==expected
        result[p.stem]=np.load(p,mmap_mode='r')
    return result


def train_backbones(spec,task,dataset,seed,training,validation,folder):
    """One common Raw/Raw-wide fit per dataset and seed, reused by all pools."""
    base,_=parents(spec);width=433
    train=(training['raw'],np.zeros((len(training['labels']),width),np.float32),training['labels'])
    val=(validation['raw'],np.zeros((len(validation['labels']),width),np.float32),validation['labels'])
    tr,va,_,stats=old._normalize_train_only(train,val,None)
    settings=dict(experiment=spec['version'],model=base['raw_model'],training=base['training'],evaluation={'test_enabled':False})
    results={};folder.mkdir(parents=True,exist_ok=False)
    for arm in ('raw','raw_wide'):
        start=time.perf_counter()
        outcome=old.train_raw(settings,dataset,arm,seed,(tr,va,None),stats,torch.device('cuda'),folder)
        results[arm]=dict(outcome=outcome,seconds=time.perf_counter()-start,checkpoint_sha256=old.sha(outcome['checkpoint']))
    write(folder/'backbones.json',dict(results=results,normalization=stats,task=task,dataset=dataset,seed=seed))
    return results


def prepare(spec,config,index):
    task,dataset=units(spec)[index];root=cache_folder(spec,task,dataset)
    root.mkdir(parents=True,exist_ok=False);files={};evidence={};data={}
    for role in ('train','validation'):
        data[role],evidence[role]=read_source(spec,task,dataset,role)
        files[role]=save_cache(root/role,data[role])
    if task=='Task4C':assert len(data['train']['labels'])==27000 and len(data['validation']['labels'])==3000
    else:train_backbones(spec,task,dataset,spec['screen_seed'],data['train'],data['validation'],root/'temporary_backbones')
    write(root/'manifest.json',dict(status='PASS',identity=identity(config),files=files,evidence=evidence,test_read=False))
    print('PREPARED',task,dataset,flush=True)


def feature_arrays(data,definition,task):
    x=data['source'];parts=[];batch=256 if task=='Task4C' else 4096
    for start in range(0,len(x),batch):
        parts.append(select_features(np.array(x[start:start+batch]),definition).numpy())
    return np.concatenate(parts)


def residual_fit(spec,config,task,dataset,seed,pool,profile,data,folder,backbones):
    base,_=parents(spec);tr,va=data['train'],data['validation'];width=433
    aux=[pad_auxiliary(feature_arrays(d,pool,task),width) for d in (tr,va)]
    normalized=old._normalize_train_only((tr['raw'],aux[0],tr['labels']),(va['raw'],aux[1],va['labels']),None)
    nt,nv,_,stats=normalized
    settings=residual_spec(base,task,profile,backbones/'checkpoints',seed)
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
        feature_dimensions=pool['feature_dimensions'],parameters=state['total_parameter_count'],
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
    train,val=data['train'],data['validation'];arrays={};mean=std=None
    for role in ('train','validation'):
        d=data[role];x=transform_fmt(feature_arrays(d,pool,'Task4C'),True);mask=np.asarray(d['mask'])
        if role=='train':
            valid=x[mask[...,0]>.5];mean=valid.mean(0,dtype=np.float64);std=valid.std(0,dtype=np.float64);std[std<1e-8]=1
        arrays[role]=np.concatenate((np.clip(((x-mean)/std).astype(np.float32),-8,8)*mask,mask),-1)
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
        normalization=dict(mean=mean.tolist(),std=std.tolist()),history=history,
        predictions_sha256=old.sha(folder/'validation_predictions.npz'),identity=identity(config))
    write(folder/'result.json',result)
    return result,(model,dict(mean=mean,std=std))


def run_directory(spec,phase,task,dataset,pool,profile,seed):
    return Path(spec['output'])/phase/task/dataset/(pool+'_'+profile)/f'seed{seed}'


def search(spec,config,index,refine=False):
    task,dataset=units(spec)[index//(3 if refine else 5)];group=index%(3 if refine else 5)
    if refine:
        short=json.loads((Path(spec['output'])/'shortlist.json').read_text())
        candidates=[(p,h) for p in short['pools'] for h in ('h1','h2','h3')][group*3:group*3+3]
    else:candidates=[(p['id'],'h0') for p in spec['pools'][group*10:group*10+10]]
    phase='refine' if refine else 'search';seed=spec['screen_task4_seed'] if task=='Task4C' else spec['screen_seed']
    data={role:read_cache(spec,task,dataset,role) for role in ('train','validation')}
    for pid,profile in candidates:
        pool=next(p for p in spec['pools'] if p['id']==pid)
        folder=run_directory(spec,phase,task,dataset,pid,profile,seed)
        if task=='Task4C':result,_=task4_fit(spec,config,seed,pool,profile,data,folder)
        else:
            result,item=residual_fit(spec,config,task,dataset,seed,pool,profile,data,folder,cache_folder(spec,task,dataset)/'temporary_backbones')
            del item;cleanup_weights(spec,folder)
        audit_validation(folder)
        print('FINISHED',phase,task,dataset,pid,profile,result['validation']['f1'],flush=True)


def audit_validation(folder):
    row=json.loads((folder/'result.json').read_text())
    assert row['test_read'] is False and old.sha(folder/'validation_predictions.npz')==row['predictions_sha256']
    with np.load(folder/'validation_predictions.npz') as z:
        y=z['labels'];p=z['probability'];t=float(z['threshold']);pred=p>=t
        tp=int(np.sum((y==1)&pred));fp=int(np.sum((y==0)&pred));fn=int(np.sum((y==1)&~pred))
        f1=2*tp/max(2*tp+fp+fn,1)
        assert abs(f1-row['validation']['f1'])<1e-12 and np.isfinite(p).all()
    return row


def ranked(spec,phases):
    rows=[]
    for phase in phases:
        for path in (Path(spec['output'])/phase).glob('*/*/*/seed*/result.json'):rows.append(audit_validation(path.parent))
    groups=collections.defaultdict(list)
    for row in rows:groups[(row['pool'],row['profile'])].append(row)
    result=[]
    for (pool,profile),items in groups.items():
        assert len(items)==21 and {(r['task'],r['dataset']) for r in items}==set(units(spec))
        scores={t:float(np.mean([r['validation']['f1'] for r in items if r['task']==t])) for t in ('Task3','Task5','Task4C')}
        result.append(dict(pool=pool,profile=profile,score=float(np.mean(list(scores.values()))),tasks=scores,
            mean_parameters=float(np.mean([r['parameters'] for r in items])),
            total_fit_seconds=sum(r['seconds'] for r in items)))
    return sorted(result,key=lambda r:(-r['score'],r['mean_parameters'],r['pool'],r['profile'])),rows


def reconcile_control(spec):
    """The p00/h0 seed reproduces frozen 8.1 before using any search outcome."""
    checks=[]
    for task,dataset in units(spec):
        seed=spec['screen_task4_seed'] if task=='Task4C' else spec['screen_seed']
        folder=run_directory(spec,'search',task,dataset,'p00','h0',seed)
        row=json.loads((folder/'result.json').read_text())
        if task=='Task4C':
            oldfolder=Path(spec['task4_reference'])/'variants/fmt_v8_100/runs'/f'regularized_fmt_mlp_seed{seed}'
            before=json.loads((oldfolder/'result.json').read_text());expected=before['validation']['pooled']['f1']
            with np.load(folder/'validation_predictions.npz') as z,np.load(oldfolder/'predictions.npz') as previous:
                assert np.array_equal(z['probability'],previous['validation_probability'])
                assert np.array_equal(z['labels'],previous['validation_labels'])
            assert row['selected_epoch']==before['selected_epoch']
        else:
            before=json.loads((Path(spec['task135_reference'])/'shards'/task/dataset/f'seed{seed}'/'frozen_models.json').read_text())['fmt_v8_100']
            expected=before['training_result']['validation_f1']
            assert row['selected_epoch']==before['best_epoch'] and row['alpha']==before['alpha']
        assert abs(row['validation']['f1']-expected)<1e-12,(task,dataset,row['validation']['f1'],expected)
        checks.append(dict(task=task,dataset=dataset,seed=seed,validation_f1=expected))
    write(Path(spec['output'])/'control_reproduction.json',dict(status='PASS',checks=checks))


def shortlist(spec,config):
    ranking,rows=ranked(spec,['search']);assert len(ranking)==50 and len(rows)==1050
    reconcile_control(spec)
    pools=[r['pool'] for r in ranking[:3]]
    write(Path(spec['output'])/'shortlist.json',dict(pools=pools,ranking=ranking,identity=identity(config),test_read=False))


def select(spec,config):
    ranking,rows=ranked(spec,['search','refine']);assert len(ranking)==59 and len(rows)==1239
    best=ranking[0];baseline=next(r for r in ranking if (r['pool'],r['profile'])==('p00','h0'))
    write(Path(spec['output'])/'selection.lock.json',dict(selected=best,baseline=baseline,ranking=ranking,
        relative_validation_gain=best['score']/baseline['score']-1,identity=identity(config),
        test_read=False,selected_at_utc=datetime.now(timezone.utc).isoformat(),
        selection_rule='Equal mean of Task3 ten-flow mean, Task5 ten-flow mean, Task4C pooled F1'))


def final(spec,config,index):
    task,dataset=units(spec)[index//3];si=index%3
    seed=(spec['final_task4_seeds'] if task=='Task4C' else spec['final_seeds'])[si]
    lockfile=Path(spec['output'])/'selection.lock.json';lock=json.loads(lockfile.read_text());assert lock['test_read'] is False
    chosen=[('p00','h0'),(lock['selected']['pool'],lock['selected']['profile'])];chosen=list(dict.fromkeys(chosen))
    data={role:read_cache(spec,task,dataset,role) for role in ('train','validation')}
    temporary=Path(spec['output'])/'final_backbones'/task/dataset/f'seed{seed}'
    if task!='Task4C':train_backbones(spec,task,dataset,seed,data['train'],data['validation'],temporary)
    fitted=[]
    for pid,profile in chosen:
        pool=next(p for p in spec['pools'] if p['id']==pid);folder=run_directory(spec,'final',task,dataset,pid,profile,seed)
        if task=='Task4C':row,item=task4_fit(spec,config,seed,pool,profile,data,folder)
        else:row,item=residual_fit(spec,config,task,dataset,seed,pool,profile,data,folder,temporary)
        fitted.append((pool,profile,folder,row,item));audit_validation(folder)
    freeze=time.time();test,evidence=read_source(spec,task,dataset,'test',allow_test=True)
    for pool,profile,folder,row,(model,state) in fitted:
        model.eval();aux=feature_arrays(test,pool,task);y=test['labels']
        if task=='Task4C':
            x=transform_fmt(aux,True);x=np.clip(((x-state['mean'])/state['std']).astype(np.float32),-8,8)
            x=np.concatenate((x*test['mask'],test['mask']),-1);probs=[]
            with torch.no_grad():
                for start in range(0,len(y),128):probs.append(model(torch.as_tensor(x[start:start+128],device='cuda')).softmax(-1)[:,1].cpu().numpy())
            probability=np.concatenate(probs)
        else:
            stats=state['normalization'];raw=((test['raw']-stats['raw_mean'])/stats['raw_std']).astype(np.float32)
            aux=((pad_auxiliary(aux,433)-stats['fmt_mean'])/stats['fmt_std']).astype(np.float32)
            yy,r,a=old._predict_components(model,old._loader((raw,aux,y),512,False,seed,True),torch.device('cuda'))
            assert np.array_equal(yy,y);probability=old._probabilities(r,a,state['alpha'],state['config']['model'])
        extras={key:test[key] for key in ('scale_id','flow_index','head_component','instance') if key in test}
        np.savez_compressed(folder/'test_predictions.npz',labels=y,probability=probability,threshold=row['threshold'],**extras)
        row.update(test=old.metrics(y,probability,row['threshold']),test_read=True,models_frozen_before_test=freeze,
            selection_lock_sha256=old.sha(lockfile),test_predictions_sha256=old.sha(folder/'test_predictions.npz'),
            test_source_evidence=evidence)
        row['per_group']={}
        for key in ('scale_id','flow_index'):
            if key in test:
                row['per_group'][key]={str(int(i)):old.metrics(y[test[key]==i],probability[test[key]==i],row['threshold']) for i in np.unique(test[key])}
        write(folder/'final_result.json',row);cleanup_weights(spec,folder)
    del fitted
    if temporary.exists():cleanup_weights(spec,temporary)
    print('FINAL',task,dataset,seed,flush=True)


def preflight(spec,config):
    assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0)
    identity(config);cleanup_weights(spec,Path(spec['output'])/'engineering_smoke')
    torch.manual_seed(1701);checks=[]
    values=torch.randn(19,138);values[0,:8]=torch.tensor([-9.,9.,-8.,7.,-6.,5.,0.,0.]);values[1]=0
    grouped=values.reshape(-1,6,23).sort(-2,descending=True).values;values=grouped.flatten(1)
    source=torch.randn(19,233);source[:,23:161]=values
    for pool in spec['pools']:
        result=pool_neighbors(values,pool);expected=pool_numpy(values.numpy(),pool)
        np.testing.assert_allclose(result.numpy(),expected,rtol=2e-6,atol=1e-6)
        assert result.shape==(19,pool['pooled_dimensions']) and not torch.any(result[1])
        selected=select_features(source,pool);assert torch.equal(selected[:,:23],source[:,:23]) and torch.equal(selected[:,23:95],source[:,161:])
        for profile in PROFILES:
            model=task4_model(pool['feature_dimensions'],profile).cuda()
            x=torch.cat((selected[:,None,:].expand(-1,10,-1),torch.ones(19,10,1)),-1).cuda()
            loss=model(x).square().mean();loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        checks.append(dict(pool=pool['id'],width=pool['feature_dimensions'],numpy_pool_check=True))
    assert torch.equal(select_features(source,spec['pools'][0]),compose_features(source[:,:161],source[:,161:]))
    # Real training-only fixture: check source identity and both task adapters.
    base,_=parents(spec);rows=old.records(base,'Task3',base['datasets'][0],'train');row=rows[0]
    raw=row['raw'][:64];cached=row['cached_fmt'][:64]
    assert np.array_equal(pathline_source(raw,cached),shared_features(raw,cached)[:,:233])
    for profile in PROFILES:
        raw_model=PathlineBinaryClassifier3D(variant='raw',fmt_dim=433,**base['raw_model'])
        settings=residual_spec(base,'Task3',profile,Path('.'),0)
        model=PathlineFMTResidualClassifier3D(raw_model,fmt_dim=433,**residual_model_kwargs(settings['model'])).cuda()
        x=torch.from_numpy(raw).cuda();a=torch.from_numpy(pad_auxiliary(select_features(pathline_source(raw,cached),spec['pools'][0]).numpy(),433)).cuda()
        output=model.forward_components(x,a);loss=sum(v.square().mean() for v in output);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    # Exercise the complete training/restoration/prediction path on disjoint
    # subsets drawn entirely from the existing training sets.
    from unittest.mock import patch
    base,b4=parents(spec);base=copy.deepcopy(base);b4=copy.deepcopy(b4)
    base['training'].update(max_epochs=1,patience=1);b4['training'].update(epochs=1,patience=1)
    raw=old.raw_array(rows);labels=old.labels(rows);cached=np.concatenate([r['cached_fmt'] for r in rows])
    indexes={c:np.flatnonzero(labels==c) for c in (0,1)}
    subsets={'train':np.r_[indexes[0][:128],indexes[1][:64]],
             'validation':np.r_[indexes[0][128:160],indexes[1][64:80]]}
    small={role:dict(raw=raw[ids],labels=labels[ids],source=pathline_source(raw[ids],cached[ids])) for role,ids in subsets.items()}
    fixture,_=read_source(spec,'Task4C','channel_tbl','train')
    indexes={c:np.flatnonzero(fixture['labels']==c) for c in (0,1)}
    small4={role:{key:value[ids] for key,value in fixture.items()} for role,ids in {
        'train':np.r_[indexes[0][:32],indexes[1][:32]],
        'validation':np.r_[indexes[0][32:48],indexes[1][32:48]]}.items()}
    smoke=Path(spec['output'])/'engineering_smoke'/('attempt'+str(spec.get('execution_revision',1)))
    with patch(__name__+'.parents',return_value=(base,b4)):
        backbones=smoke/'temporary_backbones'
        train_backbones(spec,'Task3',base['datasets'][0],0,small['train'],small['validation'],backbones)
        for profile in PROFILES:
            folder=smoke/'Task3'/profile
            _,item=residual_fit(spec,config,'Task3',base['datasets'][0],0,spec['pools'][0],profile,small,folder,backbones)
            del item;audit_validation(folder)
            folder=smoke/'Task4C'/profile
            task4_fit(spec,config,0,spec['pools'][0],profile,small4,folder);audit_validation(folder)
    cleanup_weights(spec,smoke)
    write(Path(spec['output'])/'preflight.json',dict(status='PASS',identity=identity(config),checks=checks,
        all_four_profiles_finite=True,frozen_v8_pool_exact=True,source_233_exact=True,
        real_training_only_smoke_all_profiles=True,test_read=False))


def runtime(spec,config,phase,state,code=None):
    row=dict(version=spec['version'],phase=phase,state=state,exit_code=code,
        time=datetime.now(timezone.utc).isoformat(),device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',**identity(config))
    root=Path(spec['output']);write(root/'lifecycle_events'/f"{row['job']}_{state}_{time.time_ns()}.json",row)
    append_record(root/'runtime_events.jsonl',row,'FMTv8 Search 2.1 runtime');append_record('docs/ibex_run_registry.md',row,'FMTv8 Search 2.1 runtime')


def submit(spec,config,phase,dependency=None):
    root=Path(spec['output']);logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    gpu=phase in ('preflight','prepare','search','refine','final');count={'prepare':21,'search':105,'refine':63,'final':63}.get(phase)
    cmd=['sbatch','--parsable','--cpus-per-task=4','--mem=32G','--time=02:00:00','--job-name=v8s21-'+phase,
         '--output='+str(logs/(phase+'.%A_%a.out')),'--error='+str(logs/(phase+'.%A_%a.err'))]
    if gpu:cmd+=['--gres=gpu:1','--constraint=v100']
    if count:cmd+=['--array=0-'+str(count-1)+'%12']
    if dependency:cmd+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
    cmd+=['ibex_bash/fmtv8_search_2p1.sh',phase,config]
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    row=dict(version=spec['version'],phase=phase,job_id=job,command=cmd,submitted_at_utc=datetime.now(timezone.utc).isoformat(),
        config=config,expected_device='V100' if gpu else 'CPU',**identity(config))
    append_record(root/'submissions.jsonl',row,'FMTv8 Search 2.1 submitted');append_record('docs/ibex_run_registry.md',row,'FMTv8 Search 2.1 submitted')
    print(json.dumps(row),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','prepare','search','shortlist','refine','select','final','merge','runtime','submit'])
    p.add_argument('--config',default=DEFAULT_CONFIG);p.add_argument('--index',type=int,default=0)
    p.add_argument('--submit-phase');p.add_argument('--dependency');p.add_argument('--runtime-phase');p.add_argument('--state');p.add_argument('--exit-code',type=int)
    a=p.parse_args();s=load_spec(a.config);torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    if a.phase=='runtime':runtime(s,a.config,a.runtime_phase,a.state,a.exit_code)
    elif a.phase=='submit':submit(s,a.config,a.submit_phase,a.dependency)
    elif a.phase=='prepare':prepare(s,a.config,a.index)
    elif a.phase in ('search','refine'):search(s,a.config,a.index,a.phase=='refine')
    elif a.phase=='final':final(s,a.config,a.index)
    elif a.phase=='merge':
        from experiments.Audit_FMTv8_Search_2_1 import merge
        merge(s,a.config)
    else:{'preflight':preflight,'shortlist':shortlist,'select':select}[a.phase](s,a.config)


if __name__=='__main__':main()
