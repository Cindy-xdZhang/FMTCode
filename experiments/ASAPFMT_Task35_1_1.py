"""Frozen Task3/5 data, c156 direct classifiers, discrete rigid-camera study."""
from __future__ import annotations
import argparse
import copy
import csv
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
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

from FMT_Utils import ASAPFMT_Task35_1_1 as method
from FMT_Utils.ASAPFrame_1_1 import asap_frame, transformed_observer
from FMT_Utils.GeometricControls_3D import array_hash
from experiments.Run_Task1235_AngleFeatures_1_1 import records, source_paths, evidence
from experiments.Run_Task135_GeometricControls import sources
from experiments.Verify_Task3_FMTClassifier import _select_f1_threshold

CONFIG = 'config/mainExp_ASAPFMT_Task35_1.1.json'
SOURCE_FILES = ('FMT_Utils/ASAPFrame_1_1.py', 'FMT_Utils/ASAPFMT_Task35_1_1.py',
    'FMT_Utils/Task4C_FPSAugmentSearch_1_1.py', 'FMT_Utils/DFT_FMT_3D.py',
    'FMT_Utils/FMT_P35_NormFrequency_3_1.py', 'FMT_Utils/FMT_V8_Search_2_1.py',
    'FMT_Utils/GeometricControls_3D.py', 'experiments/ASAPFMT_Task35_1_1.py',
    'experiments/Run_Task1235_AngleFeatures_1_1.py', 'experiments/Run_Task135_GeometricControls.py',
    'experiments/Verify_Task3_FMTClassifier.py', 'ibex_bash/asap_fmt_task35_1p1.sh',
    'config/mainExp_Task135_FMTv8_1.1.json', CONFIG)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def spec_load(config):
    spec=json.loads(Path(config).read_text())
    parent=json.loads(Path(spec['parent_config']).read_text())
    assert spec['arms']==list(method.ARMS) and spec['seeds']==[40,41,42]
    spec['datasets']=parent['datasets']
    return spec,parent


def identity(config):
    return dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config_sha256=sha(config),sources={name:sha(name) for name in SOURCE_FILES})


def deterministic(device):
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    if device=='cuda':
        assert torch.cuda.is_available()
        assert 'V100' in torch.cuda.get_device_name(),torch.cuda.get_device_name()


def units(spec):return [(task,dataset) for task in spec['tasks'] for dataset in spec['datasets']]


def source_evidence(parent,task,dataset,role,rows):
    result=evidence(rows)
    phase,rebuilt,_=source_paths(parent,task,dataset,role)
    if not rebuilt:
        _,label_dir,_,_=sources(parent,task,dataset,phase)
        for item,r in zip(result,rows):
            label=label_dir/Path(r['path']).name
            item.update(label_path=str(label),label_sha256=sha(label))
    return result


def encode_rows(rows,arm,device,batch):
    chunks=[];started=time.perf_counter()
    for r in rows:
        for first in range(0,len(r['raw']),batch):
            sl=slice(first,first+batch)
            chunks.append(method.encode(r['raw'][sl],r['times'][sl],arm,device).cpu().numpy())
    return np.concatenate(chunks),time.perf_counter()-started


def identifiers(rows):
    return dict(labels=np.concatenate([r['labels'] for r in rows]).astype(np.int64),
        scale_id=np.concatenate([r['scale_id'] for r in rows]),
        row_id=np.concatenate([np.column_stack([np.full(len(r['raw']),r['ordinal']),np.arange(len(r['raw']))]) for r in rows]))


def fit_stats(values):return method.normalizer(values)


def metrics(y,p,threshold):
    predicted=p>=threshold
    return dict(f1=float(f1_score(y,predicted,zero_division=0)),
        precision=float(precision_score(y,predicted,zero_division=0)),
        recall=float(recall_score(y,predicted,zero_division=0)),
        average_precision=float(average_precision_score(y,p)) if np.any(y) else 0.,
        accuracy=float(np.mean(y==predicted)),sample_count=len(y),positives=int(y.sum()),
        confusion_matrix=confusion_matrix(y,predicted,labels=[0,1]).tolist())


def scale_metrics(ids,p,threshold):
    return {str(k):metrics(ids['labels'][ids['scale_id']==k],p[ids['scale_id']==k],threshold)
            for k in np.unique(ids['scale_id'])}


def objectivity_check(x,t,device):
    x=np.asarray(x,dtype=np.float64);changed=transformed_observer(x)
    y,detail=asap_frame(x,t,return_details=True);other=asap_frame(changed,t)
    radius=np.linalg.norm(x-x[:,:1],axis=-1).max((1,2))
    relative_error=float(np.max(np.abs(y-other)/radius[:,None,None,None]))
    assert relative_error<2e-7,relative_error
    a=method.encode(x,t,'asap_fmt',device).cpu().numpy()
    b=method.encode(changed,t,'asap_fmt',device).cpu().numpy()
    np.testing.assert_allclose(a,b,atol=2e-4,rtol=2e-4)
    independent=method.encode(x[:1],t[:1],'asap_fmt',device).cpu().numpy()
    np.testing.assert_allclose(independent,a[:1],atol=2e-5,rtol=2e-5)
    return dict(samples=len(x),coordinate_error_over_radius=relative_error,
        feature_absolute_error=float(np.max(np.abs(a-b))),minimum_rank_ratio=detail['minimum_rank_ratio'],
        batch_independence=True,observer='independent proper rotation per original time sample plus translation')


def preflight(spec,parent,config):
    deterministic('cuda')
    from tests.test_asap_frame_1_1 import ASAPTests
    import unittest
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ASAPTests)
    outcome=unittest.TextTestRunner(verbosity=2).run(suite);assert outcome.wasSuccessful()
    torch.set_num_threads(4)
    rows=records(parent,'Task3','channel','train');r=rows[0]
    selected=np.r_[np.flatnonzero(r['labels']==0)[:16],np.flatnonzero(r['labels']==1)[:16]]
    assert len(selected)==32
    x,t=r['raw'][selected],r['times'][selected]
    y=torch.tensor(r['labels'][selected],device='cuda',dtype=torch.float32)
    checks=objectivity_check(x,t,'cuda');pilots={}
    for arm in spec['arms']:
        torch.manual_seed(910917)
        encoded=method.encode(x,t,arm,'cuda').cpu().numpy()
        features=torch.tensor(method.standardize(encoded,fit_stats(encoded)),device='cuda')
        model=method.classifier(arm).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        losses=[]
        for step in range(160):
            model.train();optimizer.zero_grad(set_to_none=True)
            out=model(features);loss=F.binary_cross_entropy_with_logits(out[:,1]-out[:,0],y)
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():p=model(features).softmax(-1)[:,1].cpu().numpy()
        score=metrics(y.cpu().numpy(),p,.5)
        assert score['f1']>=.95 and losses[-1]<losses[0],(arm,score,losses[0],losses[-1])
        # A full-size backward step and finite gradients verify the real architecture.
        model.train();model.zero_grad(set_to_none=True)
        big=features.repeat(16,1,1);model(big).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        pilots[arm]=dict(training_f1=score['f1'],first_loss=losses[0],last_loss=losses[-1],
            parameters=sum(p.numel() for p in model.parameters()),batch512_backward=True)
    write(Path(spec['output'])/'preflight.json',dict(status='PASS',identity=identity(config),
        objectivity=checks,pilots=pilots,gpu=torch.cuda.get_device_name(),test_read=False,
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))


def prepare(spec,parent,config,index):
    deterministic('cpu');task,dataset=units(spec)[index]
    root=Path(spec['output']);assert json.loads((root/'preflight.json').read_text())['status']=='PASS'
    folder=root/'cache'/task/dataset;folder.mkdir(parents=True,exist_ok=False)
    manifest=dict(task=task,dataset=dataset,identity=identity(config),roles={},test_read=False)
    norm={}
    for role in ('train','validation'):
        rows=records(parent,task,dataset,role);target=folder/role;target.mkdir()
        ids=identifiers(rows);files={}
        for name,value in ids.items():
            path=target/(name+'.npy');np.save(path,value);files[name+'.npy']=sha(path)
        report=dict(samples=len(ids['labels']),positives=int(ids['labels'].sum()),
                    source=source_evidence(parent,task,dataset,role,rows),files=files,encoding_seconds={})
        if task=='Task5':
            expected='train' if role=='train' else 'validation'
            assert all(r['metadata']['scale_set']==expected for r in rows)
            report['scale_table']=rows[0]['metadata']['scale_table']
        if role=='train':
            r=rows[0];take=np.r_[np.flatnonzero(r['labels']==0)[:16],np.flatnonzero(r['labels']==1)[:16]]
            report['objectivity']=objectivity_check(r['raw'][take],r['times'][take],'cpu')
            begin=time.perf_counter();asap_frame(r['raw'][:256],r['times'][:256])
            report['camera_seconds_per_primitive']=(time.perf_counter()-begin)/min(256,len(r['raw']))
        for arm in spec['arms']:
            values,elapsed=encode_rows(rows,arm,'cpu',spec['execution']['encode_batch'])
            if role=='train':norm[arm]=fit_stats(values)
            path=target/(arm+'.npy');np.save(path,method.standardize(values,norm[arm]));files[path.name]=sha(path)
            report['encoding_seconds'][arm]=elapsed
        manifest['roles'][role]=report
    train_paths={r['path'] for r in manifest['roles']['train']['source']}
    val_paths={r['path'] for r in manifest['roles']['validation']['source']}
    assert not train_paths&val_paths
    write(folder/'normalization.json',norm);manifest['normalization_sha256']=sha(folder/'normalization.json')
    manifest['status']='PASS';write(folder/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('task','dataset','status')}),flush=True)


@torch.no_grad()
def predict(model,x,batch):
    model.eval();return torch.cat([model(x[i:i+batch]).softmax(-1)[:,1] for i in range(0,len(x),batch)]).cpu().numpy()


def train_arm(spec,arm,seed,x,y,vx,vy,folder):
    settings=spec['training'];torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=method.classifier(arm).cuda();optimizer=torch.optim.AdamW(model.parameters(),
        lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    weight=torch.tensor((len(y)-float(y.sum()))/float(y.sum()),device='cuda')
    target=torch.tensor(y,device='cuda',dtype=torch.float32)
    best=-np.inf;selected_epoch=0;state=None;history=[];started=time.perf_counter()
    for epoch in range(1,settings['max_epochs']+1):
        order=rng.permutation(len(y));model.train();total=0.;batch=settings['batch_size']
        for first in range(0,len(order),batch):
            idx=torch.tensor(order[first:first+batch],device='cuda')
            optimizer.zero_grad(set_to_none=True);logits=model(x[idx])
            loss=F.binary_cross_entropy_with_logits(logits[:,1]-logits[:,0],target[idx],pos_weight=weight)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(idx)
        probability=predict(model,vx,batch);score=float(average_precision_score(vy,probability))
        improved=score>best+settings['min_delta']
        if improved:
            best=score;selected_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        row=dict(epoch=epoch,training_loss=total/len(y),validation_average_precision=score,
            selected=improved,permutation_sha256=array_hash(order),seconds=time.perf_counter()-started)
        history.append(row)
        with (folder/(arm+'_history.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
        if epoch==1 or epoch%10==0:print(arm,seed,row,flush=True)
        if epoch-selected_epoch>=settings['patience']:break
    assert state is not None
    model.load_state_dict(state);p=predict(model,vx,batch)
    threshold=float(_select_f1_threshold(vy,p))
    result=dict(arm=arm,seed=seed,parameters=sum(v.numel() for v in model.parameters()),
        selected_epoch=selected_epoch,epochs=len(history),validation=metrics(vy,p,threshold),
        threshold=threshold,training_seconds=time.perf_counter()-started)
    return state,result,p


def save_prediction(folder,name,ids,p,threshold):
    path=folder/(name+'.npz');np.savez_compressed(path,**ids,probability=p,threshold=threshold)
    return sha(path)


def train(spec,parent,config,index):
    deterministic('cuda');task,dataset=units(spec)[index//3];seed=spec['seeds'][index%3]
    root=Path(spec['output']);cache=root/'cache'/task/dataset
    manifest=json.loads((cache/'manifest.json').read_text());assert manifest['status']=='PASS'
    assert manifest['identity']==identity(config)
    for role in ('train','validation'):
        for name,digest in manifest['roles'][role]['files'].items():assert sha(cache/role/name)==digest
    assert sha(cache/'normalization.json')==manifest['normalization_sha256']
    norm=json.loads((cache/'normalization.json').read_text())
    folder=root/'runs'/task/dataset/('seed'+str(seed));folder.mkdir(parents=True,exist_ok=False)
    ids={role:{k:np.load(cache/role/(k+'.npy')) for k in ('labels','scale_id','row_id')} for role in ('train','validation')}
    models={};results={}
    for arm in spec['arms']:
        x=torch.tensor(np.load(cache/'train'/(arm+'.npy')),device='cuda')
        vx=torch.tensor(np.load(cache/'validation'/(arm+'.npy')),device='cuda')
        state,result,p=train_arm(spec,arm,seed,x,ids['train']['labels'],vx,ids['validation']['labels'],folder)
        result['predictions']={'validation':save_prediction(folder,arm+'_validation',ids['validation'],p,result['threshold'])}
        result['validation_by_scale']=scale_metrics(ids['validation'],p,result['threshold'])
        models[arm]=state;results[arm]=result
        del x,vx;torch.cuda.empty_cache()
    lock=dict(identity=identity(config),task=task,dataset=dataset,seed=seed,test_loaded=False,
        arms={arm:{k:r[k] for k in ('selected_epoch','threshold','parameters')} for arm,r in results.items()},
        normalization_sha256=manifest['normalization_sha256'],locked_at=datetime.now(timezone.utc).isoformat())
    write(folder/'selection.lock.json',lock)
    evaluations=[('test',task)]
    if task=='Task3':evaluations.append(('transfer_task5','Task5'))
    source_records={}
    for name,evaluation_task in evaluations:
        rows=records(parent,evaluation_task,dataset,'confirmation');test_ids=identifiers(rows)
        source_records[name]=source_evidence(parent,evaluation_task,dataset,'confirmation',rows)
        for arm in spec['arms']:
            values,elapsed=encode_rows(rows,arm,'cuda',spec['execution']['encode_batch'])
            x=torch.tensor(method.standardize(values,norm[arm]),device='cuda')
            model=method.classifier(arm).cuda();model.load_state_dict(models[arm]);p=predict(model,x,spec['training']['batch_size'])
            result=results[arm];result[name]=metrics(test_ids['labels'],p,result['threshold'])
            result[name+'_by_scale']=scale_metrics(test_ids,p,result['threshold'])
            result[name+'_encoding_seconds']=elapsed
            result['predictions'][name]=save_prediction(folder,arm+'_'+name,test_ids,p,result['threshold'])
            del x,model;torch.cuda.empty_cache()
    output=dict(status='PASS',version=spec['version'],identity=identity(config),task=task,dataset=dataset,seed=seed,
        gpu=torch.cuda.get_device_name(),node=socket.gethostname(),source=source_records,
        selection_lock_sha256=sha(folder/'selection.lock.json'),results=results,checkpoint_files_created=False)
    write(folder/'result.json',output)
    print('FINAL',task,dataset,seed,json.dumps({a:r['test']['f1'] for a,r in results.items()}),flush=True)


def merge(spec,parent,config):
    root=Path(spec['output']);rows=[];source_count=set();predictions=0
    for task,dataset in units(spec):
        expected_ids={'validation':identifiers(records(parent,task,dataset,'validation')),
                      'test':identifiers(records(parent,task,dataset,'confirmation'))}
        if task=='Task3':expected_ids['transfer_task5']=identifiers(records(parent,'Task5',dataset,'confirmation'))
        for seed in spec['seeds']:
            folder=root/'runs'/task/dataset/('seed'+str(seed));result=json.loads((folder/'result.json').read_text())
            assert result['status']=='PASS' and result['identity']==identity(config)
            assert sha(folder/'selection.lock.json')==result['selection_lock_sha256']
            lock=json.loads((folder/'selection.lock.json').read_text());assert not lock['test_loaded']
            for arm,r in result['results'].items():
                history=[json.loads(line) for line in (folder/(arm+'_history.jsonl')).read_text().splitlines()]
                best=-np.inf;chosen=0
                for item in history:
                    if item['validation_average_precision']>best+spec['training']['min_delta']:
                        chosen=item['epoch'];best=item['validation_average_precision']
                assert chosen==r['selected_epoch']==lock['arms'][arm]['selected_epoch']
                for role,digest in r['predictions'].items():
                    path=folder/(arm+'_'+role+'.npz');assert sha(path)==digest
                    with np.load(path) as z:
                        for key in ('labels','row_id','scale_id'):
                            assert np.array_equal(z[key],expected_ids[role][key]),(task,dataset,seed,arm,role,key)
                        score=metrics(z['labels'],z['probability'],float(z['threshold']))
                        assert score==r['validation' if role=='validation' else role]
                        assert float(z['threshold'])==lock['arms'][arm]['threshold']
                        if role=='validation':assert float(z['threshold'])==float(_select_f1_threshold(z['labels'],z['probability']))
                    predictions+=1
                rows.append(dict(task=task,dataset=dataset,seed=seed,arm=arm,parameters=r['parameters'],
                    selected_epoch=r['selected_epoch'],epochs=r['epochs'],threshold=r['threshold'],
                    training_seconds=r['training_seconds'],**{k:r['test'][k] for k in ('f1','average_precision','precision','recall','accuracy')}))
            for sources_ in result['source'].values():
                for item in sources_:
                    for key,digestkey in (('path','sha256'),('label_path','label_sha256')):
                        if key in item and item[key] not in source_count:
                            assert sha(item[key])==item[digestkey];source_count.add(item[key])
    assert len(rows)==180 and predictions==450
    with (root/'per_run.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={};table=['# ASAP FMT Task3/5 — 1.1','', '| Task | Method | F1 mean ± sample SD | AP mean ± sample SD |', '|---|---|---:|---:|']
    for task in spec['tasks']:
        summary[task]={}
        for arm in spec['arms']:
            subset=[r for r in rows if r['task']==task and r['arm']==arm]
            means={key:[float(np.mean([r[key] for r in subset if r['seed']==s])) for s in spec['seeds']]
                   for key in ('f1','average_precision')}
            record={key:dict(seeds=values,mean=float(np.mean(values)),std=float(np.std(values,ddof=1))) for key,values in means.items()}
            record['per_dataset']={d:{key:dict(mean=float(np.mean([r[key] for r in subset if r['dataset']==d])),std=float(np.std([r[key] for r in subset if r['dataset']==d],ddof=1))) for key in means} for d in spec['datasets']}
            summary[task][arm]=record
            table.append(f"| {task} | {arm} | {record['f1']['mean']:.6f} ± {record['f1']['std']:.6f} | {record['average_precision']['mean']:.6f} ± {record['average_precision']['std']:.6f} |")
        summary[task]['paired_asap_minus_fmt']={key:[a-b for a,b in zip(summary[task]['asap_fmt'][key]['seeds'],summary[task]['fmt_c156'][key]['seeds'])] for key in ('f1','average_precision')}
    transfer={}
    for arm in spec['arms']:
        transfer[arm]={}
        for key in ('f1','average_precision'):
            values=[]
            for seed in spec['seeds']:
                scores=[json.loads((root/'runs'/'Task3'/d/('seed'+str(seed))/'result.json').read_text())['results'][arm]['transfer_task5'][key] for d in spec['datasets']]
                values.append(float(np.mean(scores)))
            transfer[arm][key]=dict(seeds=values,mean=float(np.mean(values)),std=float(np.std(values,ddof=1)))
    summary['fixed_task3_transfer_to_task5']=transfer
    write(root/'summary.json',dict(status='PASS',identity=identity(config),predictions_verified=predictions,
        source_files_rechecked=len(source_count),training_runs=len(rows),summary=summary))
    (root/'report.md').write_text('\n'.join(table)+'\n')
    print(json.dumps(summary),flush=True)


def runtime(spec,config,phase,state,code):
    folder=Path(spec['output'])/'runtime';folder.mkdir(parents=True,exist_ok=True)
    job=os.environ.get('SLURM_JOB_ID','local');array=os.environ.get('SLURM_ARRAY_TASK_ID','none')
    row=dict(version=spec['version'],phase=phase,state=state,exit_code=code,job_id=job,array_index=array,
        time_utc=datetime.now(timezone.utc).isoformat(),node=socket.gethostname(),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',identity=identity(config))
    write(folder/(job+'_'+array+'_'+state+'.json'),row)


def submit(spec,config):
    root=Path(spec['output']);(root/'logs').mkdir(parents=True,exist_ok=True);dependency=None;submitted=[]
    assert not (root/'submissions.json').exists()
    for phase,count,gpu in [('preflight',1,True),('prepare',20,False),('train',60,True),('merge',1,False)]:
        cmd=['sbatch','--parsable','--job-name=asap35-'+phase,'--cpus-per-task=4','--mem=12G',
            '--time='+('02:00:00' if phase=='train' else '00:40:00'),
            '--output='+str(root/'logs'/(phase+'.%A_%a.out')),'--error='+str(root/'logs'/(phase+'.%A_%a.err'))]
        if gpu:cmd+=['--gres=gpu:1','--constraint=v100']
        if count>1:cmd+=['--array=0-'+str(count-1)+'%'+str(spec['execution']['max_parallel_gpus'] if gpu else spec['execution']['prepare_parallel_cpus'])]
        if dependency:cmd+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
        cmd+=['ibex_bash/asap_fmt_task35_1p1.sh',phase,config]
        job=subprocess.check_output(cmd,text=True).strip().split(';')[0];dependency=job
        item=dict(phase=phase,job_id=job,processes=count,command=cmd,time_utc=datetime.now(timezone.utc).isoformat(),identity=identity(config))
        submitted.append(item);write(root/'submissions.json',submitted)
        with Path('docs/ibex_run_registry.md').open('a') as f:
            f.write('\nASAPFMT Task35 1.1 submission: '+json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['preflight','prepare','train','merge','runtime','submit'])
    parser.add_argument('--config',default=CONFIG);parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--runtime-phase');parser.add_argument('--state');parser.add_argument('--exit-code',type=int)
    args=parser.parse_args();spec,parent=spec_load(args.config)
    if args.phase=='runtime':runtime(spec,args.config,args.runtime_phase,args.state,args.exit_code)
    elif args.phase=='submit':submit(spec,args.config)
    elif args.phase=='preflight':preflight(spec,parent,args.config)
    elif args.phase=='prepare':prepare(spec,parent,args.config,args.index)
    elif args.phase=='train':train(spec,parent,args.config,args.index)
    else:merge(spec,parent,args.config)


if __name__=='__main__':main()
