"""Task4-c full-training-set memorization; no validation or test reads."""
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
from FMT_Utils.Task4C_LinePooling_4_3 import make_model as frozen_model
from FMT_Utils.Task4C_Multiscale_4_1 import transform_fmt

DEFAULT_CONFIG = 'config/Verify_Task4C_TrainingMemorization_4.12.json'


def load_spec(config):
    definition = json.loads(Path(config).read_text())
    if pipeline.sha(definition['base_config']) != definition['base_config_sha256']:
        raise ValueError('Frozen base config changed')
    if 'flows' in definition:
        return definition
    base = json.loads(Path(definition['base_config']).read_text())
    spec = {k:base[k] for k in ('flows','cache_source','scale_sets','encoding','parent_output')}
    spec.update(definition)
    return spec


def identity(config):
    result = pipeline.identity(config)
    for name in ('experiments/Task4C_TrainingMemorization_4_12.py',
                 'ibex_bash/task4c_training_memorization_4p12.sh'):
        result['source_sha256'][name] = pipeline.sha(name)
    definition = json.loads(Path(config).read_text())
    result['source_sha256'][definition['base_config']] = pipeline.sha(definition['base_config'])
    return result


def encode(spec, config):
    source = Path(spec['cache_source']['output'])
    if pipeline.sha(source/'encoding.json') != spec['cache_source']['encoding_sha256']:
        raise ValueError('Frozen cache manifest changed')
    original = json.loads((source/'encoding.json').read_text())
    if original['identity']['config_sha256'] != spec['cache_source']['config_sha256']:
        raise ValueError('Frozen cache config changed')
    for name, expected in original['identity']['source_sha256'].items():
        if pipeline.sha(name) != expected:
            raise ValueError('Frozen source changed: '+name)
    report = dict(version=spec['version'],identity=identity(config),splits={},
                  validation_encoded=False,test_encoded=False)
    for flow in spec['flows']:
        key = flow['name']+'/train'
        entry = original['splits'][key]
        destination = Path(spec['output'])/key
        destination.mkdir(parents=True,exist_ok=False)
        for name in ('fmt.npy','voxels.npy','metadata.npz'):
            if pipeline.sha(source/key/name) != entry['files'][name]:
                raise ValueError('Source cache changed')
            shutil.copyfile(source/key/name,destination/name)
            if pipeline.sha(destination/name) != entry['files'][name]:
                raise ValueError('Cache copy failed')
        report['splits'][key] = copy.deepcopy(entry)
    if sum(x['samples'] for x in report['splits'].values()) != spec['expected_training_samples']:
        raise ValueError('Must fit the entire prescribed training set')
    pipeline.write(Path(spec['output'])/'encoding.json',report)


def load_data(spec, split, method, manifest):
    if split != 'train':
        raise ValueError('Memorization only reads original training data')
    candidate = dict(architecture='learned_pool_wider_conv_4.3')
    return pipeline.load_data(spec,split,method,manifest,candidate)


class WideLinePooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.line = nn.Sequential(nn.Linear(233,512),nn.GELU(),nn.Linear(512,512),nn.GELU())
        self.head = nn.Sequential(nn.Linear(1024,1024),nn.GELU(),nn.Linear(1024,512),nn.GELU(),nn.Linear(512,2))

    def forward(self,x):
        mask = x[...,-1]>.5
        z = self.line(x[...,:-1])
        mean = (z*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
        maximum = z.masked_fill(~mask[...,None],-torch.inf).amax(1)
        return self.head(torch.cat((mean,maximum),-1))


class WideConv3D(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for i,(cin,cout) in enumerate(((4,32),(32,64),(64,128))):
            layers.extend((nn.Conv3d(cin,cout,3,padding=1),nn.GroupNorm(4,cout),nn.GELU(),
                           nn.MaxPool3d(2) if i<2 else nn.AdaptiveAvgPool3d(2)))
        self.network=nn.Sequential(*layers,nn.Flatten(),nn.Linear(1024,1024),nn.GELU(),
                                   nn.Linear(1024,512),nn.GELU(),nn.Linear(512,2))

    def forward(self,x):
        return self.network(x)


def make_model(method,capacity):
    if method not in ('fmt_mlp','conv3d_mlp'):
        raise ValueError('Unknown method')
    if capacity=='original':
        return frozen_model(method,dict(architecture='learned_pool_wider_conv_4.3',dropout=0.))
    if capacity!='wide':
        raise ValueError('Unknown capacity')
    return WideLinePooling() if method=='fmt_mlp' else WideConv3D()


def plan(spec):
    return [(method,capacity) for capacity in spec['capacities'] for method in spec['methods']]


def folder(spec,method,capacity):
    return Path(spec['output'])/'runs'/f'{capacity}_{method}'


def train(spec,config,index):
    method,capacity=plan(spec)[index]
    out=folder(spec,method,capacity);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((Path(spec['output'])/'encoding.json').read_text())
    if manifest['identity']['config_sha256']!=pipeline.sha(config):
        raise ValueError('Cache identity mismatch')
    options=spec['training'];seed=options['seed']
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.environ.get('SLURM_JOB_ID') and device.type!='cuda':
        raise RuntimeError('Allocated training GPU unavailable')
    started=time.time()
    data=load_data(spec,'train',method,manifest)
    n=len(data['labels']);assert n==spec['expected_training_samples']
    normalization=None
    if method=='fmt_mlp':
        mask=data['features'][...,-1:]
        x=transform_fmt(data['features'][...,:-1],True)
        valid=x[mask[...,0]>.5]
        mean,std=valid.mean(0,dtype=np.float64),valid.std(0,dtype=np.float64)
        std[std<1e-8]=1
        x=np.clip(((x-mean)/std).astype(np.float32),-8.,8.)
        data['features']=np.concatenate((x*mask,mask),-1)
        normalization=dict(mean=mean.tolist(),std=std.tolist(),clip=8.,signed_log=True)
    model=make_model(method,capacity).to(device)
    assert not any(isinstance(m,(nn.Dropout,nn.Dropout3d)) and m.p!=0 for m in model.modules())
    optimizer=torch.optim.Adam(model.parameters(),lr=options['learning_rate'],weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,
        patience=options['lr_patience'],threshold=1e-4,min_lr=options['minimum_learning_rate'])
    batch=options['batch_size']
    def predict():
        model.eval();prob=[];total_loss=0.
        with torch.no_grad():
            for start in range(0,n,batch):
                x=torch.as_tensor(data['features'][start:start+batch],device=device,dtype=torch.float32)
                y=torch.as_tensor(data['labels'][start:start+batch],device=device,dtype=torch.long)
                logits=model(x)
                total_loss+=float(F.cross_entropy(logits,y,reduction='sum'))
                prob.append(logits.softmax(-1)[:,1].cpu().numpy())
        return np.concatenate(prob),total_loss/n
    best_score=(-1.,-np.inf);best=None;state=None;history=[];streak=0
    for epoch in range(1,options['epochs']+1):
        order=rng.permutation(n)
        if not np.array_equal(np.sort(order),np.arange(n)):
            raise ValueError('Training permutation missed or repeated samples')
        lr=optimizer.param_groups[0]['lr'];model.train();online=0.
        for start in range(0,n,batch):
            ids=order[start:start+batch]
            x=torch.as_tensor(data['features'][ids],device=device,dtype=torch.float32)
            y=torch.as_tensor(data['labels'][ids],device=device,dtype=torch.long)
            optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),options['gradient_clip'],error_if_nonfinite=True);optimizer.step()
            online+=float(loss.detach())*len(ids)
        probability,loss=predict();metrics=pipeline.binary_metrics(data['labels'],probability,.5)
        errors=int(np.sum((probability>=.5)!=data['labels']))
        streak=streak+1 if metrics['f1']>=spec['target']['f1'] else 0
        row=dict(epoch=epoch,learning_rate=lr,online_loss=online/n,full_training_loss=loss,
                 training_f1=metrics['f1'],errors=errors,training_examples=n,
                 permutation_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
                 seconds=time.time()-started,consecutive_passes=streak)
        history.append(row)
        with (out/'history.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(row)+'\n')
        pipeline.write(out/'progress.json',dict(method=method,capacity=capacity,**row))
        if epoch==1 or epoch%10==0 or streak:
            print(json.dumps(dict(method=method,capacity=capacity,**row)),flush=True)
        score=(metrics['f1'],-loss)
        if score>best_score:
            best_score=score;best=dict(epoch=epoch,probability=probability.copy(),metrics=metrics,errors=errors,loss=loss)
            state=copy.deepcopy(model.state_dict())
        scheduler.step(loss)
        if streak>=spec['target']['consecutive_epochs']:break
    model.load_state_dict(state);probability,loss=predict()
    if not np.array_equal(probability,best['probability']):raise ValueError('Best-state restore mismatch')
    threshold=pipeline.choose_threshold(data['labels'],probability)
    np.savez_compressed(out/'predictions.npz',training_probability=probability,training_labels=data['labels'],
        training_flow_index=data['flow_index'],training_head_component=data['head_component'],training_scale_id=data['scale_id'])
    result=dict(version=spec['version'],identity=identity(config),method=method,capacity=capacity,seed=seed,
        training_examples=n,parameters=sum(p.numel() for p in model.parameters()),selected_epoch=best['epoch'],
        threshold=.5,training=pipeline.all_metrics(data,probability,.5,spec),errors=best['errors'],
        training_loss=loss,training_selected_threshold=threshold,
        training_selected_threshold_metrics=pipeline.binary_metrics(data['labels'],probability,threshold),
        target_reached=streak>=spec['target']['consecutive_epochs'],history=history,normalization=normalization,
        validation_loaded=False,test_loaded=False,weights_written=0,predictions_sha256=pipeline.sha(out/'predictions.npz'),
        device=torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',torch=torch.__version__,
        started_at_utc=datetime.fromtimestamp(started,timezone.utc).isoformat(),
        finished_at_utc=datetime.now(timezone.utc).isoformat(),seconds=time.time()-started)
    pipeline.write(out/'result.json',result)


def merge(spec,config):
    records=[]
    for method,capacity in plan(spec):
        p=folder(spec,method,capacity);r=json.loads((p/'result.json').read_text())
        if r['identity']['config_sha256']!=pipeline.sha(config) or r['predictions_sha256']!=pipeline.sha(p/'predictions.npz'):
            raise ValueError('Result identity changed')
        records.append(r)
    selected={method:max([r for r in records if r['method']==method],key=lambda r:(r['target_reached'],r['training']['pooled']['f1'])) for method in spec['methods']}
    summary=dict(version=spec['version'],identity=identity(config),
        rows=[{k:r[k] for k in ('method','capacity','parameters','selected_epoch','training_examples','training','errors','target_reached')} for r in records],
        both_models_reached_target=all(r['target_reached'] for r in selected.values()),
        validation_loaded=False,test_loaded=False)
    pipeline.write(Path(spec['output'])/'summary.json',summary)
    print(json.dumps(summary),flush=True)


def runtime(spec,config,phase,state,code=None):
    row=dict(time=datetime.now(timezone.utc).isoformat(),phase=phase,state=state,exit_code=code,
             device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',**identity(config))
    pipeline.append_locked(Path(spec['output'])/'runtime_events.jsonl',json.dumps(row)+'\n')
    pipeline.append_locked('docs/ibex_run_registry.md','\n- Task4-c 4.12 runtime '+json.dumps(row)+'\n')


def submit(spec,config):
    out=Path(spec['output']).resolve();out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir()
    pipeline.write(out/'config.frozen.json',spec);previous=None
    for phase in ('encode','train','merge'):
        gpu=phase=='train'
        command=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=32G',
                 '--time=06:00:00' if gpu else '--time=00:15:00',f'--job-name=t4c412-{phase}',
                 f'--output={out}/logs/{phase}_%A_%a.out',f'--error={out}/logs/{phase}_%A_%a.err']
        if previous:command.extend((f'--dependency=afterok:{previous}','--kill-on-invalid-dep=yes'))
        if gpu:command.extend(('--gres=gpu:1','--constraint=a100|v100',f'--array=0-{len(plan(spec))-1}%4'))
        command.extend(('ibex_bash/task4c_training_memorization_4p12.sh',phase,config))
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row=dict(job_id=job,phase=phase,version=spec['version'],submitted_at_utc=datetime.now(timezone.utc).isoformat(),
                 command=command,dependency=previous,expected_device='A100/V100' if gpu else 'CPU',**identity(config))
        pipeline.append_locked(out/'submissions.jsonl',json.dumps(row)+'\n')
        pipeline.append_locked('docs/ibex_run_registry.md','\n- Task4-c 4.12 SUBMITTED '+json.dumps(row)+'\n')
        print(json.dumps(row),flush=True);previous=job


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('encode','train','merge','submit','runtime'))
    p.add_argument('--config',default=DEFAULT_CONFIG);p.add_argument('--index',type=int,default=0)
    p.add_argument('--runtime-phase');p.add_argument('--state');p.add_argument('--exit-code',type=int)
    a=p.parse_args();spec=load_spec(a.config)
    if a.phase=='train':train(spec,a.config,a.index)
    elif a.phase=='runtime':runtime(spec,a.config,a.runtime_phase,a.state,a.exit_code)
    else:{'encode':encode,'merge':merge,'submit':submit}[a.phase](spec,a.config)


if __name__=='__main__':main()
