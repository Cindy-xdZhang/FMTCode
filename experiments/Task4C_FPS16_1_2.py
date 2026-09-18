"""Fixed original center with FPS16 on the rebuilt >=17-line bundle set; paired six-neighbour controls (1.2).

The encoders, networks, training rules and the four arms are exactly those of
``Ablation_Task4C_FPS16_1.1``; only the data rebuild rules changed (see
``FMT_Utils/Task4C_FPS16_Data_1_2.py`` and ``docs/Task4C_FPS16_protocol_1.2.md``).
"""
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
from FMT_Utils import Task4C_FPS16_Data_1_2 as data
from experiments import Task4C_FPS16_1_1 as base
from experiments import Task4C_OriginalCenter_1_1 as previous

frozen,p35=previous.frozen,previous.p35
sha,write=frozen.sha,frozen.write
CONFIG='config/Ablation_Task4C_FPS16_1.2.json'
FILES=base.FILES+('FMT_Utils/Task4C_FPS16_Data_1_2.py','experiments/Task4C_FPS16_1_2.py',
    'FMT_Utils/Task4C_Multiscale_4_1.py','FMT_Utils/Task4C_HairpinBinary_2_1.py','config/Ablation_Task4C_FPS16_1.1.json')


def identity(config):
    return dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                config_sha256=sha(config),sources={p:sha(p) for p in FILES})


def verify_source(spec,config):
    """Read-only check of the frozen template and the rows that must be replaced."""
    checked=[];report={};minimum=spec['replacement']['minimum_valid_lines']
    for flow in spec['flows']:
        name=flow['name'];report[name]={}
        for role in data.ROLES:
            folder=Path(spec['source_output'])/'physical'/name/role
            for filename,digest in spec['source_files'][name][role]['files'].items():
                assert sha(folder/filename)==digest,(name,role,filename);checked.append(str(folder/filename))
            m=data.metadata(folder);assert len(m['labels'])==spec['source_files'][name][role]['samples']
            anchor,bad=data.failing_rows(np.load(folder/'seeds.npy',mmap_mode='r'),m,minimum)
            report[name][role]=dict(samples=len(anchor),missing_center=int((anchor<0).sum()),fewer_than_minimum=int((m['counts']<minimum).sum()),
                replace=int(bad.sum()),replace_positive=int((bad&(m['labels']==1)).sum()),replace_gt_head=int((bad&(m['head_component']<0)).sum()),
                retained_center_in_slot0=bool(np.all(anchor[~bad]==0)))
    assert {role:sum(v[role]['samples'] for v in report.values()) for role in data.ROLES}==spec['expected_counts']
    write(Path(spec['output'])/'source_verification.json',dict(complete=True,identity=identity(config),flows=report,files=checked,
          minimum_valid_lines=minimum,template='complete_GT_head_coverage_rows'))


class Dataset:
    def __init__(self,spec,role,candidate,device='cuda',limit=None):
        self.device=device;self.parts=[];self.role=role
        if role=='test':
            lock=Path(spec['output'])/'final'/candidate['id']/f"seed{spec['_active_seed']}"/'selection.lock.json'
            assert json.loads(lock.read_text())['test_loaded'] is False
        self.evidence=None if limit else Path(spec['output'])/'final'/candidate['id']/f"seed{spec['_active_seed']}"
        labels=[];owners=[];flows=[];rows=[];source_rows=[];centers=[];replaced=[];relaxed=[]
        root=Path(spec['_root_output']);minimum=spec['replacement']['minimum_valid_lines']
        for fi,flow in enumerate(spec['flows']):
            name=flow['name']
            if limit:
                # Engineering pilot on retained source rows only; never a performance claim.
                folder=Path(spec['source_output'])/'physical'/name/role;m=data.metadata(folder)
                anchor,bad=data.failing_rows(np.load(folder/'seeds.npy',mmap_mode='r'),m,minimum)
                ids=np.concatenate([np.flatnonzero((m['labels']==cls)&~bad)[:limit//4] for cls in (0,1)])
                anchor=anchor[ids];original_rows=ids;changed=np.zeros(len(ids),bool);loose=np.zeros(len(ids),bool)
            else:
                folder=root/'physical'/name/role;m=data.metadata(folder);ids=np.arange(len(m['labels']))
                with np.load(folder/'replacement_index.npz') as z:
                    anchor=z['original_center_id'];original_rows=z['source_row'];changed=z['replaced'];loose=z['relaxed']
            assert np.all(anchor>=0) and np.all(m['counts'][ids]>=minimum)
            self.parts.append((folder,ids,m['counts'][ids],anchor))
            labels.append(m['labels'][ids]);owners.append(m['instance'][ids]);flows.append(np.full(len(ids),fi));rows.append(ids)
            source_rows.append(original_rows);centers.append(anchor);replaced.append(changed);relaxed.append(loose)
        self.labels=np.concatenate(labels).astype(np.int64);self.owners=np.concatenate(owners);self.flows=np.concatenate(flows)
        self.rows=np.concatenate(rows);self.source_rows=np.concatenate(source_rows);self.anchor=np.concatenate(centers)
        self.replaced=np.concatenate(replaced);self.relaxed=np.concatenate(relaxed)
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
                center_policy='one_original_center_only',neighbors=candidate['neighbors'],replaced_rows=int(self.replaced.sum()),relaxed_rows=int(self.relaxed.sum()),
                center_ids_sha256=hashlib.sha256(self.anchor.astype('<i8').tobytes()).hexdigest(),
                neighbor_ids_sha256=hashlib.sha256(self.selected_neighbors.astype('<i8').tobytes()).hexdigest()))


def arm_spec(spec,index):
    arm=copy.deepcopy(spec);arm['_root_output']=spec['output'];arm['candidate']=spec['candidates'][index]
    arm['output']=str(Path(spec['output'])/'arms'/arm['candidate']['id']);arm['_active_seed']=spec['final']['seeds'][0]
    return arm


def save_predictions(folder,role,dataset,probability):
    file=folder/f'{role}_predictions.npz'
    np.savez_compressed(file,labels=dataset.labels,probability=probability,instance=dataset.owners,flow_index=dataset.flows,
        row_in_split=dataset.rows,source_row=dataset.source_rows,replaced=dataset.replaced,relaxed=dataset.relaxed,
        original_center_id=dataset.anchor,threshold=.5)
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
    """Independent recomputation of every rebuilt file against the frozen template and the 1.2 rules."""
    root=Path(spec['output']);totals={role:0 for role in data.ROLES};reports={};minimum=spec['replacement']['minimum_valid_lines']
    for flow in spec['flows']:
        name=flow['name'];reports[name]={};metas={};indices={}
        build=json.loads((root/'physical'/name/'preparation.json').read_text());assert build['complete'] and build['identity']==identity(config) and not build['pilot']
        for role in data.ROLES:
            folder=root/'physical'/name/role;original=Path(spec['source_output'])/'physical'/name/role
            for filename,digest in spec['source_files'][name][role]['files'].items():assert sha(original/filename)==digest
            for filename,digest in build['reports'][role]['files'].items():assert sha(folder/filename)==digest
            m=data.metadata(folder);before=data.metadata(original);metas[role]=m
            with np.load(folder/'replacement_index.npz') as z:idx={k:z[k] for k in z.files}
            indices[role]=idx;ids=idx['source_row'];assert np.array_equal(ids,np.arange(len(before['labels'])))
            old_anchor,failing=data.failing_rows(np.load(original/'seeds.npy',mmap_mode='r'),before,minimum)
            replaced=idx['replaced'];relaxed=idx['relaxed'];assert np.all(failing<=replaced) and np.all(relaxed<=replaced)
            extra=replaced&~failing
            if role=='train':assert not extra.any()
            for key in ('labels','instance'):assert np.array_equal(m[key][~relaxed],before[key][~relaxed]),key
            assert np.array_equal(m['scale_id'],before['scale_id']) and np.array_equal(m['head_component']<0,before['head_component']<0)
            assert np.array_equal(idx['source_label'],before['labels']) and np.array_equal(idx['source_instance'],before['instance'])
            assert len(np.unique(m['center'],axis=0))==len(m['center'])
            g=np.load(folder/'geometry.npy',mmap_mode='r');s=np.load(folder/'seeds.npy',mmap_mode='r')
            center,ratio=method.original.original_center_indices(s,m)
            assert np.array_equal(center,idx['original_center_id']) and np.all(center>=0) and np.all(m['counts']>=minimum)
            assert np.all(center[replaced]==0) and np.all(idx['neighbor_filter'][replaced]>0) and np.all(idx['neighbor_filter'][~replaced]==0)
            assert np.array_equal(idx['neighbor_filter']==data.FILTER_RELAXED,relaxed)
            mask=np.arange(27)[None]<m['counts'][:,None]
            relative=m['half_arc_lengths']/m['requested_half_length'][:,None,None]
            assert np.all(relative[mask]>=.95) and np.all(relative[mask]<=1.002)
            for filename,new in (('geometry.npy',g),('seeds.npy',s)):
                old=np.load(original/filename,mmap_mode='r')
                for start in range(0,len(ids),512):
                    take=np.arange(start,min(start+512,len(ids)));keep=take[~replaced[take]]
                    assert np.array_equal(new[keep],old[keep]);assert np.isfinite(new[take]).all()
                    block=np.asarray(new[take],np.float64)
                    if filename=='geometry.npy':assert np.all(block[~mask[take]]==0)
            assert np.all(np.linalg.norm(m['center'][replaced]-before['center'][replaced],axis=1)>1e-10)
            assert np.array_equal(m['center'][~replaced],before['center'][~replaced])
            totals[role]+=len(ids);reports[name][role]=dict(samples=len(ids),replaced=int(replaced.sum()),retained=int((~replaced).sum()),
                failing_template_rows=int(failing.sum()),replaced_for_lost_same_head_train=int(extra.sum()),relaxed=int(relaxed.sum()),
                label_changes=int(np.sum(m['labels']!=before['labels'])),maximum_center_error_over_spacing=float(ratio.max()),
                minimum_lines=int(m['counts'].min()),line_count_histogram=np.bincount(m['counts'],minlength=28)[minimum:].tolist(),
                classes=np.bincount(m['labels'],minlength=2).tolist(),files=build['reports'][role]['files'])
        training=metas['train'];tree=cKDTree(training['center']);v_tree=cKDTree(metas['validation']['center'])
        heads={int(h):cKDTree(training['center'][training['head_component']==h]) for h in np.unique(training['head_component'])}
        for role in ('validation','test'):
            m=metas[role];distance=tree.query(m['center'])[0]
            assert np.all(distance>=m['local_grid_scale']-1e-12)
            assert np.allclose(distance,m['nearest_train_center_distance'],rtol=1e-12,atol=1e-12)
            for head in np.unique(m['head_component']):
                take=m['head_component']==head;same=heads[int(head)].query(m['center'][take])[0]
                assert np.all(same<=4*m['local_grid_scale'][take]+1e-12)
                assert np.allclose(same,m['nearest_same_head_train_center_distance'][take],rtol=1e-12,atol=1e-12)
            if role=='test':assert np.all(v_tree.query(m['center'])[0]>=.5*m['local_grid_scale']-1e-12)
        for role in ('train','test'):
            keep=~indices[role]['relaxed'];before=data.metadata(Path(spec['source_output'])/'physical'/name/role)
            assert np.array_equal(np.unique(metas[role]['instance'][keep]),np.unique(before['instance'][keep]))
        # Independent recomputation of the per-line seed attributes from the frozen fields.
        physical_spec=json.loads(Path(spec['physical_config']).read_text());fi=[f['name'] for f in physical_spec['flows']].index(name)
        scene=data.coverage.load_scene(physical_spec,fi);threshold=float(scene['flow']['lambda2_threshold'])
        for role in data.ROLES:
            folder=root/'physical'/name/role;m=metas[role];idx=indices[role]
            with np.load(folder/data.ATTRIBUTE_FILE) as z:attr={k:z[k] for k in z.files}
            assert float(attr['lambda2_threshold'])==threshold
            valid=np.arange(27)[None]<m['counts'][:,None];points=attr['seed_points']
            assert np.array_equal(np.isfinite(points).all(-1),valid) and np.array_equal(attr['exact_stencil_coordinates'],idx['replaced'])
            recon=data.reconstructed_seeds(np.load(folder/'seeds.npy',mmap_mode='r'),m)
            error=np.linalg.norm(np.nan_to_num(points-recon),axis=-1)/m['neighbor_distance'][:,None]
            assert error[valid].max()<1e-3
            for i in np.flatnonzero(idx['replaced'])[:2000]:
                grid=data.stencil(m['center'][i],m['neighbor_distance'][i]);n=int(m['counts'][i])
                assert np.array_equal(points[i,:n],grid[attr['stencil_slot'][i,:n]]) and attr['stencil_slot'][i,0]==0
            fresh=data.line_attributes(points,scene['axes'],scene['lambda2'],scene['oyf'],threshold)
            for key in ('lambda2','oyf'):assert np.allclose(np.nan_to_num(fresh[key]),np.nan_to_num(attr[key]),atol=1e-6,rtol=1e-6)
            for key in ('inside','head_candidate','oyf_positive'):assert np.array_equal(fresh[key],attr[key])
            assert not np.any(attr['head_candidate']&~valid) and not np.any(attr['oyf_positive']&~valid)
            assert np.all(attr['head_candidate'][idx['replaced']&~idx['relaxed'],0])
            reports[name][role]['lines']=dict(total=int(valid.sum()),head_candidate=int(attr['head_candidate'].sum()),
                oyf_positive=int(attr['oyf_positive'].sum()),retained_lines_not_head_candidate=int((valid&~idx['replaced'][:,None]&~attr['head_candidate']).sum()),
                relaxed_centers_not_head_candidate=int((idx['relaxed']&~attr['head_candidate'][:,0]).sum()))
        del scene
    assert totals==spec['expected_counts']
    write(root/'data_audit.json',dict(complete=True,identity=identity(config),counts=totals,flows=reports,
        original_center_required=True,minimum_valid_lines=minimum,source_unchanged=True,retained_geometry_exact=True,
        neighbor_filter_recorded_per_row=True,per_line_attributes_recomputed=True))


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
            key=np.column_stack([pred[k] for k in ('labels','instance','flow_index','row_in_split','source_row','replaced','relaxed','original_center_id')])
            if role in paired:assert np.array_equal(key,paired[role])
            else:paired[role]=key
            for fi,flow in enumerate(spec['flows']):
                mask=pred['flow_index']==fi;folder_data=root/'physical'/flow['name']/role;m=data.metadata(folder_data)
                with np.load(folder_data/'replacement_index.npz') as z:
                    assert np.array_equal(pred['original_center_id'][mask],z['original_center_id'])
                    assert np.array_equal(pred['replaced'][mask],z['replaced']) and np.array_equal(pred['relaxed'][mask],z['relaxed'])
                assert np.array_equal(pred['row_in_split'][mask],np.arange(len(m['labels'])))
                assert np.array_equal(pred['labels'][mask],m['labels']) and np.array_equal(pred['instance'][mask],m['instance'])
            y=pred['labels'];p=pred['probability'];assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
            score=dict(f1=float(f1_score(y,p>=.5)),precision=float(precision_score(y,p>=.5)),recall=float(recall_score(y,p>=.5)))
            expected=r['validation'] if role=='validation' else r['test']['combined']
            for k,v in score.items():assert abs(v-expected[k])<1e-12
            subsets=dict(retained=~pred['replaced'],replaced=pred['replaced'],relaxed=pred['relaxed'])
            scores[role]=dict(combined=frozen.metrics(y,p),
                per_flow={flow['name']:frozen.metrics(y[pred['flow_index']==fi],p[pred['flow_index']==fi]) for fi,flow in enumerate(spec['flows'])},
                **{k:(frozen.metrics(y[v],p[v]) if v.any() else None) for k,v in subsets.items()})
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
    phases=[('verify-source',[],None,False,'00:20:00','32G'),('gpu-check',['verify-source'],None,True,'00:15:00','32G'),
            ('pilot',['verify-source'],'0-1',False,'03:00:00','64G'),('build',['pilot','gpu-check'],'0-1',False,'12:00:00','64G'),
            ('audit-data',['build'],None,False,'01:00:00','48G'),('train',['audit-data'],'0-3%4',True,'06:00:00','64G'),
            ('audit-results',['train'],None,False,'00:20:00','16G')]
    for phase,dependencies,array,gpu,wall,memory in phases:
        command=['sbatch','--parsable','--propagate=NONE','--job-name=FPS16v12_'+phase,'--cpus-per-task=4','--mem='+memory,'--time='+wall,
            '--output='+str(root/'logs/%x_%A_%a.out'),'--error='+str(root/'logs/%x_%A_%a.err')]
        if dependencies:command+=['--dependency=afterok:'+':'.join(jobs[d] for d in dependencies)]
        if array:command+=['--array='+array]
        if gpu:command+=['--gres=gpu:1','--constraint=v100','--exclude=gpu213-18']
        command+=['ibex_bash/task4c_fps16_1p2.sh',phase,config]
        jobs[phase]=subprocess.check_output(command,text=True).strip().split(';')[0]
        frozen.append_locked(root/'submissions.jsonl',json.dumps(dict(phase=phase,job=jobs[phase],command=command,
            submitted_at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
        write(root/'submission.json',dict(identity=identity(config),jobs=jobs))
    print(json.dumps(jobs),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['verify-source','gpu-check','pilot','build','audit-data','train','audit-results','submit','runtime'])
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
