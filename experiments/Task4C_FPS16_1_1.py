"""Fixed original center with FPS16, fresh replacement bundles and paired controls."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from types import FunctionType,SimpleNamespace
import numpy as np
import torch
from scipy.spatial import cKDTree
from FMT_Utils import Task4C_FPS16_1_1 as method
from FMT_Utils import Task4C_FPS16_Data_1_1 as data
from experiments import Task4C_OriginalCenter_1_1 as previous

frozen,p35=previous.frozen,previous.p35
sha,write=frozen.sha,frozen.write
CONFIG='config/Ablation_Task4C_FPS16_1.1.json'
FILES=previous.FILES+('FMT_Utils/Task4C_FPS16_1_1.py','FMT_Utils/Task4C_FPS16_Data_1_1.py',
    'experiments/Task4C_FPS16_1_1.py','experiments/Task4C_PhysicalLength_4_14.py',
    'FMT_Utils/Task4C_GTHeadCoverage_1_1.py','FMT_Utils/Task4C_InstanceCoverage_9_15.py','FMT_Utils/Task4C_BottomDensity_1_2.py',
    'FMT_Utils/Task4C_BottomDensity_1_1.py','FMT_Utils/Task4C_Bundles_1_1.py','FMT_Utils/Task4C_PaperBundles_3_1.py',
    'config/mainExp_Task4C_GTHeadCoverage_1.1.json','config/Ablation_Task4C_BottomDensity_1.2.json')


def identity(config):
    return dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                config_sha256=sha(config),sources={p:sha(p) for p in FILES})


def verify_source(spec,config):
    checked=[];report={}
    for flow in spec['flows']:
        name=flow['name'];report[name]={}
        for role in data.ROLES:
            folder=Path(spec['source_output'])/'physical'/name/role
            for filename,digest in spec['source_files'][name][role]['files'].items():
                assert sha(folder/filename)==digest;checked.append(str(folder/filename))
            file=Path(spec['subset_source'])/f'{name}_{role}.npz';assert sha(file)==spec['subset_hashes'][name][role]
            with np.load(file) as z:ids=z['row_ids'];anchor=z['center_ids']
            m=data.metadata(folder);actual,_=method.original.original_center_indices(np.load(folder/'seeds.npy',mmap_mode='r'),m)
            assert np.array_equal(anchor,actual[ids]) and np.all(anchor>=0)
            report[name][role]=dict(samples=len(ids),replace=int(np.sum(m['counts'][ids]<17)))
    assert {role:sum(v[role]['samples'] for v in report.values()) for role in data.ROLES}==spec['expected_counts']
    write(Path(spec['output'])/'source_verification.json',dict(complete=True,identity=identity(config),flows=report,files=checked))


class Dataset:
    def __init__(self,spec,role,candidate,device='cuda',limit=None):
        self.device=device;self.parts=[];self.role=role
        if role=='test':
            lock=Path(spec['output'])/'final'/candidate['id']/f"seed{spec['_active_seed']}"/'selection.lock.json'
            assert json.loads(lock.read_text())['test_loaded'] is False
        self.evidence=None if limit else Path(spec['output'])/'final'/candidate['id']/f"seed{spec['_active_seed']}"
        labels=[];owners=[];flows=[];rows=[];source_rows=[];centers=[];replaced=[]
        root=Path(spec['_root_output'])
        for fi,flow in enumerate(spec['flows']):
            name=flow['name']
            if limit:
                folder=Path(spec['source_output'])/'physical'/name/role;m=data.metadata(folder)
                with np.load(Path(spec['subset_source'])/f'{name}_{role}.npz') as z:available=z['row_ids'];aa=z['center_ids']
                chosen=np.concatenate([np.flatnonzero((m['labels'][available]==cls)&(m['counts'][available]>=17))[:limit//4] for cls in (0,1)])
                ids=available[chosen];anchor=aa[chosen];original_rows=ids;changed=np.zeros(len(ids),bool)
            else:
                folder=root/'physical'/name/role;m=data.metadata(folder);ids=np.arange(len(m['labels']))
                with np.load(folder/'replacement_index.npz') as z:anchor=z['original_center_id'];original_rows=z['source_row'];changed=z['replaced']
            assert np.all(anchor>=0) and np.all(m['counts'][ids]>=17)
            self.parts.append((folder,ids,m['counts'][ids],anchor))
            labels.append(m['labels'][ids]);owners.append(m['instance'][ids]);flows.append(np.full(len(ids),fi));rows.append(ids)
            source_rows.append(original_rows);centers.append(anchor);replaced.append(changed)
        self.labels=np.concatenate(labels).astype(np.int64);self.owners=np.concatenate(owners);self.flows=np.concatenate(flows)
        self.rows=np.concatenate(rows);self.source_rows=np.concatenate(source_rows);self.anchor=np.concatenate(centers);self.replaced=np.concatenate(replaced)
        assert limit or len(self.labels)==spec['expected_counts'][role]
        self.targets=torch.tensor(self.labels,device=device);self.neighbors=torch.empty((len(self.labels),0),device=device,dtype=torch.long)
        self.clean=None

    def encode(self,candidate,batch=128):
        self.clean=torch.empty((len(self.labels),1,142),device=self.device);offset=0;neighbors=[]
        for folder,rows,counts,anchor in self.parts:
            g=np.load(folder/'geometry.npy',mmap_mode='r');s=np.load(folder/'seeds.npy',mmap_mode='r')
            for first in range(0,len(rows),batch):
                ids=rows[first:first+batch]
                tokens,n=method.encode(torch.tensor(np.array(g[ids]),device=self.device),
                    torch.tensor(np.array(s[ids]),device=self.device),torch.tensor(counts[first:first+batch],device=self.device,dtype=torch.long),
                    torch.tensor(anchor[first:first+batch],device=self.device,dtype=torch.long),candidate)
                if candidate['base_method']=='p35_h0':
                    values=tokens.cpu().numpy();values[...,:141]=p35.transform_fmt(values[...,:141],True);tokens=torch.from_numpy(values).to(self.device)
                self.clean[offset:offset+len(ids)]=tokens;offset+=len(ids);neighbors.append(n.cpu().numpy())
        assert offset==len(self.labels) and torch.isfinite(self.clean).all();self.selected_neighbors=np.concatenate(neighbors)
        assert self.selected_neighbors.shape==(len(self.labels),candidate['neighbors'])
        assert np.all(self.selected_neighbors!=self.anchor[:,None])
        assert all(len(np.unique(row))==candidate['neighbors'] for row in self.selected_neighbors)
        if self.evidence:
            write(self.evidence/f'{self.role}_encoding.json',dict(candidate=candidate,shape=list(self.clean.shape),samples=len(self.labels),
                center_policy='one_original_center_only',neighbors=candidate['neighbors'],
                center_ids_sha256=hashlib.sha256(self.anchor.astype('<i8').tobytes()).hexdigest(),
                neighbor_ids_sha256=hashlib.sha256(self.selected_neighbors.astype('<i8').tobytes()).hexdigest()))


def arm_spec(spec,index):
    arm=copy.deepcopy(spec);arm['_root_output']=spec['output'];arm['candidate']=spec['candidates'][index]
    arm['output']=str(Path(spec['output'])/'arms'/arm['candidate']['id']);arm['_active_seed']=spec['final']['seeds'][0]
    return arm


def save_predictions(folder,role,dataset,probability):
    file=folder/f'{role}_predictions.npz'
    np.savez_compressed(file,labels=dataset.labels,probability=probability,instance=dataset.owners,flow_index=dataset.flows,
        row_in_split=dataset.rows,source_row=dataset.source_rows,replaced=dataset.replaced,original_center_id=dataset.anchor,threshold=.5)
    return sha(file)


def environment(candidate):
    standardize=p35.standardize if candidate['base_method']=='p35_h0' else frozen.standardize
    predict=FunctionType(frozen.predict.__wrapped__.__code__,dict(frozen.__dict__,standardize=standardize),argdefs=frozen.predict.__wrapped__.__defaults__)
    constructor=lambda pool,architecture,profile:method.make_model(candidate['base_method'])
    return dict(frozen.__dict__,Dataset=Dataset,identity=identity,fit_normalizer=previous.fit_normalizer,
        method=SimpleNamespace(**dict(frozen.method.__dict__,FourierClassifier=constructor)),
        standardize=standardize,predict=torch.no_grad()(predict),save_predictions=save_predictions)


def gpu_check(spec,config):
    frozen.engine.deterministic('cuda');assert 'V100' in torch.cuda.get_device_name();torch.set_num_threads(4)
    checks={}
    for index,c in enumerate(spec['candidates']):
        torch.manual_seed(97811);arm=arm_spec(spec,index);d=Dataset(arm,'train',c,limit=32);d.encode(c)
        small=Dataset(arm,'train',c,limit=32);small.encode(c,batch=8)
        assert np.array_equal(d.selected_neighbors,small.selected_neighbors)
        torch.testing.assert_close(d.clean,small.clean,atol=2e-4,rtol=2e-4)
        env=environment(c);norm=previous.fit_normalizer(d,c);xx=env['standardize'](d.clean,norm)
        model=method.make_model(c['base_method']).cuda();optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        initial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest();losses=[]
        for step in range(150):
            model.train();optim.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(xx),d.targets)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optim.step();losses.append(float(loss.detach()))
        probability,_=env['predict'](model,d,norm,128);score=frozen.metrics(d.labels,probability)
        assert score['f1']>=.95 and losses[-1]<losses[0]
        optim.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(model(xx.repeat(4,1,1)),d.targets.repeat(4)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        checks[c['id']]=dict(parameters=sum(p.numel() for p in model.parameters()),shape=list(d.clean.shape),neighbors=c['neighbors'],
            pilot_f1=score['f1'],initial_loss=losses[0],final_loss=losses[-1],batch128_backward=True,
            initialization_sha256=initial,network=str(model),modules=sorted(set(type(m).__name__ for m in model.modules())))
    for name in ('p35_h0','c156'):
        assert len({checks[c['id']]['initialization_sha256'] for c in spec['candidates'] if c['base_method']==name})==1
    write(Path(spec['output'])/'gpu_check.json',dict(complete=True,identity=identity(config),gpu=torch.cuda.get_device_name(),checks=checks,engineering_only=True))


def train(spec,config,index):
    for filename in ('gpu_check.json','data_audit.json'):
        result=json.loads((Path(spec['output'])/filename).read_text());assert result['complete'] and result['identity']==identity(config)
    arm=arm_spec(spec,index);write(Path(arm['output'])/'selection.lock.json',dict(complete=True,selected=arm['candidate'],identity=identity(config)))
    env=environment(arm['candidate']);FunctionType(frozen.train.__code__,env,argdefs=frozen.train.__defaults__)(arm,config,0,'final')


def audit_data(spec,config):
    root=Path(spec['output']);totals={role:0 for role in data.ROLES};reports={}
    for flow in spec['flows']:
        name=flow['name'];reports[name]={};metas={}
        build=json.loads((root/'physical'/name/'preparation.json').read_text());assert build['complete'] and build['identity']==identity(config)
        for role in data.ROLES:
            folder=root/'physical'/name/role;original=Path(spec['source_output'])/'physical'/name/role
            for filename,digest in spec['source_files'][name][role]['files'].items():assert sha(original/filename)==digest
            for filename,digest in build['reports'][role]['files'].items():assert sha(folder/filename)==digest
            m=data.metadata(folder);before=data.metadata(original);metas[role]=m
            with np.load(folder/'replacement_index.npz') as z:idx={k:z[k] for k in z.files}
            with np.load(Path(spec['subset_source'])/f'{name}_{role}.npz') as z:assert np.array_equal(idx['source_row'],z['row_ids'])
            ids=idx['source_row'];bad=before['counts'][ids]<17;assert np.array_equal(idx['replaced'],bad)
            for key in ('labels','instance','scale_id'):assert np.array_equal(m[key],before[key][ids])
            assert np.array_equal(m['head_component']<0,before['head_component'][ids]<0)
            assert len(np.unique(m['center'],axis=0))==len(m['center'])
            g=np.load(folder/'geometry.npy',mmap_mode='r');s=np.load(folder/'seeds.npy',mmap_mode='r')
            center,ratio=method.original.original_center_indices(s,m)
            assert np.array_equal(center,idx['original_center_id']) and np.all(center>=0) and np.all(m['counts']>=17)
            mask=np.arange(27)[None]<m['counts'][:,None]
            relative=m['half_arc_lengths']/m['requested_half_length'][:,None,None]
            assert np.all(relative[mask]>=.95) and np.all(relative[mask]<=1.002)
            for filename,new in (('geometry.npy',g),('seeds.npy',s)):
                old=np.load(original/filename,mmap_mode='r')
                for start in range(0,len(ids),512):
                    take=np.arange(start,min(start+512,len(ids)));keep=take[~bad[take]]
                    assert np.array_equal(new[keep],old[ids[keep]])
                    assert np.isfinite(new[take]).all()
            assert np.all(np.linalg.norm(m['center'][bad]-before['center'][ids[bad]],axis=1)>1e-10)
            assert np.array_equal(m['center'][~bad],before['center'][ids[~bad]])
            totals[role]+=len(ids);reports[name][role]=dict(samples=len(ids),replaced=int(bad.sum()),retained=int((~bad).sum()),
                maximum_center_error_over_spacing=float(ratio.max()),minimum_lines=int(m['counts'].min()),
                classes=np.bincount(m['labels'],minlength=2).tolist(),files=build['reports'][role]['files'])
        training=metas['train'];tree=cKDTree(training['center']);v_tree=cKDTree(metas['validation']['center'])
        heads={int(h):cKDTree(training['center'][training['head_component']==h]) for h in np.unique(training['head_component'])}
        for role in ('validation','test'):
            m=metas[role];distance=tree.query(m['center'])[0]
            assert np.all(distance>=m['local_grid_scale']-1e-12)
            assert np.allclose(distance,m['nearest_train_center_distance'],rtol=1e-12,atol=1e-12)
            for head in np.unique(m['head_component']):
                take=m['head_component']==head;assert np.all(heads[int(head)].query(m['center'][take])[0]<=4*m['local_grid_scale'][take]+1e-12)
            if role=='test':assert np.all(v_tree.query(m['center'])[0]>=.5*m['local_grid_scale']-1e-12)
        for role in ('train','test'):
            expected=np.unique(data.metadata(Path(spec['source_output'])/'physical'/name/role)['instance'])
            assert np.array_equal(np.unique(metas[role]['instance']),expected)
    assert totals==spec['expected_counts']
    write(root/'data_audit.json',dict(complete=True,identity=identity(config),counts=totals,flows=reports,
        original_center_required=True,minimum_valid_lines=17,source_unchanged=True,retained_geometry_exact=True))


def audit_results(spec,config):
    root=Path(spec['output']);data_report=json.loads((root/'data_audit.json').read_text());assert data_report['complete']
    records=[];paired={};histories=[]
    from sklearn.metrics import f1_score,precision_score,recall_score
    for index,c in enumerate(spec['candidates']):
        arm=arm_spec(spec,index);folder=Path(arm['output'])/'final'/c['id']/f"seed{arm['_active_seed']}"
        r=json.loads((folder/'result.json').read_text());assert r['complete'] and r['identity']==identity(config) and r['test_loaded']
        assert r['candidate']==c and r['parameters']==c['parameters'];scores={}
        history=r['history'];histories.append([h['permutation_sha256'] for h in history]);assert all(h['samples']==spec['expected_counts']['train'] for h in history)
        best=max(history,key=lambda h:(h['validation_f1'],h['validation_average_precision']));assert best['epoch']==r['selected_epoch']
        lock=json.loads((folder/'selection.lock.json').read_text());assert lock['test_loaded'] is False and lock['selected_epoch']==r['selected_epoch']
        for role in ('validation','test'):
            file=folder/f'{role}_predictions.npz';assert sha(file)==r['predictions'][role]
            with np.load(file) as z:pred={k:z[k] for k in z.files}
            assert len(pred['labels'])==spec['expected_counts'][role] and float(pred['threshold'])==.5
            key=np.column_stack([pred[k] for k in ('labels','instance','flow_index','row_in_split','source_row','replaced','original_center_id')])
            if role in paired:assert np.array_equal(key,paired[role])
            else:paired[role]=key
            for fi,flow in enumerate(spec['flows']):
                mask=pred['flow_index']==fi;folder_data=root/'physical'/flow['name']/role;m=data.metadata(folder_data)
                with np.load(folder_data/'replacement_index.npz') as z:assert np.array_equal(pred['original_center_id'][mask],z['original_center_id'])
                assert np.array_equal(pred['row_in_split'][mask],np.arange(len(m['labels'])))
                assert np.array_equal(pred['labels'][mask],m['labels']) and np.array_equal(pred['instance'][mask],m['instance'])
            y=pred['labels'];p=pred['probability'];assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
            score=dict(f1=float(f1_score(y,p>=.5)),precision=float(precision_score(y,p>=.5)),recall=float(recall_score(y,p>=.5)))
            expected=r['validation'] if role=='validation' else r['test']['combined']
            for k,v in score.items():assert abs(v-expected[k])<1e-12
            scores[role]=dict(combined=frozen.metrics(y,p),per_flow={flow['name']:frozen.metrics(y[pred['flow_index']==fi],p[pred['flow_index']==fi]) for fi,flow in enumerate(spec['flows'])},
                retained=frozen.metrics(y[~pred['replaced']],p[~pred['replaced']]),replaced=frozen.metrics(y[pred['replaced']],p[pred['replaced']]))
        assert r['normalization']['fit_tokens']==spec['expected_counts']['train']
        records.append(dict(candidate=c,seed=r['seed'],parameters=r['parameters'],epochs=r['epochs'],selected_epoch=r['selected_epoch'],
            training_seconds=r['training_seconds'],preparation_seconds=r['preparation_seconds'],scores=scores,result_sha256=sha(folder/'result.json')))
    for h in histories[1:]:
        n=min(len(h),len(histories[0]));assert h[:n]==histories[0][:n]
    assert not list((root/'arms').rglob('*.pt')) and not list((root/'arms').rglob('*.pth'))
    write(root/'summary.json',dict(complete=True,identity=identity(config),counts=spec['expected_counts'],methods=records,
        single_seed=True,all_centers_fixed=True,paired_rows_verified=True,no_weight_files=True))


def runtime(spec,config,phase,state,code):
    frozen.append_locked(Path(spec['output'])/'runtime.jsonl',json.dumps(dict(identity=identity(config),phase=phase,state=state,exit_code=code,
        job_id=os.environ.get('SLURM_JOB_ID'),array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),array_index=os.environ.get('SLURM_ARRAY_TASK_ID'),
        hostname=socket.gethostname(),at_utc=datetime.now(timezone.utc).isoformat()))+'\n')


def submit(spec,config):
    root=Path(spec['output']);(root/'logs').mkdir(parents=True,exist_ok=True);assert not (root/'submission.json').exists();jobs={}
    phases=[('verify-source',[],None,False,'00:15:00','16G'),('gpu-check',['verify-source'],None,True,'00:10:00','32G'),
            ('pilot',['verify-source'],'0-1',False,'02:00:00','64G'),('build',['pilot','gpu-check'],'0-1',False,'12:00:00','64G'),
            ('audit-data',['build'],None,False,'00:30:00','32G'),('train',['audit-data'],'0-3%4',True,'06:00:00','64G'),
            ('audit-results',['train'],None,False,'00:15:00','16G')]
    for phase,dependencies,array,gpu,wall,memory in phases:
        command=['sbatch','--parsable','--propagate=NONE','--job-name=FPS16_'+phase,'--cpus-per-task=4','--mem='+memory,'--time='+wall,
            '--output='+str(root/'logs/%x_%A_%a.out'),'--error='+str(root/'logs/%x_%A_%a.err')]
        if dependencies:command+=['--dependency=afterok:'+':'.join(jobs[d] for d in dependencies)]
        if array:command+=['--array='+array]
        if gpu:command+=['--gres=gpu:1','--constraint=v100','--exclude=gpu213-18']
        command+=['ibex_bash/task4c_fps16_1p1.sh',phase,config]
        jobs[phase]=subprocess.check_output(command,text=True).strip().split(';')[0]
        frozen.append_locked(root/'submissions.jsonl',json.dumps(dict(phase=phase,job=jobs[phase],command=command))+'\n')
        write(root/'submission.json',dict(identity=identity(config),jobs=jobs))
    print(json.dumps(jobs),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['verify-source','gpu-check','pilot','build','audit-data','train','audit-results','submit','runtime'])
    p.add_argument('--config',default=CONFIG);p.add_argument('--index',type=int,default=0);p.add_argument('--runtime-phase');p.add_argument('--state');p.add_argument('--exit-code',type=int)
    a=p.parse_args();spec=json.loads(Path(a.config).read_text())
    if a.phase=='verify-source':verify_source(spec,a.config)
    elif a.phase=='gpu-check':gpu_check(spec,a.config)
    elif a.phase in ('pilot','build'):data.prepare(spec,a.index,write,sha,lambda:identity(a.config),pilot=a.phase=='pilot')
    elif a.phase=='audit-data':audit_data(spec,a.config)
    elif a.phase=='train':train(spec,a.config,a.index)
    elif a.phase=='audit-results':audit_results(spec,a.config)
    elif a.phase=='submit':submit(spec,a.config)
    else:runtime(spec,a.config,a.runtime_phase,a.state,a.exit_code)


if __name__=='__main__':main()
