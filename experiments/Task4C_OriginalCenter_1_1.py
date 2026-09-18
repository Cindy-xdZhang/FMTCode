"""Fixed original-seed Task4-c comparison; frozen heads and training rules."""
from __future__ import annotations
import argparse
import copy
import hashlib
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket
import subprocess
from types import FunctionType
import numpy as np
import torch
from FMT_Utils import Task4C_OriginalCenter_1_1 as method
from experiments import Task4C_FPSAugmentSearch_1_1 as frozen
from experiments import Task4C_P35GTHead_1_1 as p35

CONFIG='config/Ablation_Task4C_OriginalCenter_1.1.json'
FILES=('FMT_Utils/Task4C_OriginalCenter_1_1.py','experiments/Task4C_OriginalCenter_1_1.py',
       'FMT_Utils/Task4C_FPSAugmentSearch_1_1.py','experiments/Task4C_FPSAugmentSearch_1_1.py',
       'FMT_Utils/DFT_FMT_3D.py','FMT_Utils/FMT_P35_NormFrequency_3_1.py',
       'FMT_Utils/Task4C_LinePooling_4_3.py','FMT_Utils/FMT_V8_Search_2_1.py',
       'experiments/Task4C_P35GTHead_1_1.py','FMT_Utils/FMTNoConvolution_1_1.py')
sha,write=frozen.sha,frozen.write


def rotating(candidate):
    return candidate.get('center_policy','original_only')=='rotating_control'


def encode_candidate(g,s,c,a,candidate):
    if not rotating(candidate):
        return method.encode(g,s,c,a,candidate['base_method'])
    if candidate['base_method']=='p35_h0':
        return p35.encode_batch(g,s,c),method.select_neighbors(s,c,a,'nearest6')
    neighbors=frozen.method.neighbor_indices(s,c,'fps6')
    return (frozen.method.fourier_tokens(frozen.method.resample(g,48,'uniform'),c,neighbors,'p35'),
            neighbors[torch.arange(len(g),device=g.device),a])


def make_candidate_model(candidate):
    if not rotating(candidate):return method.make_model(candidate['base_method'])
    model=frozen.method.FourierClassifier('p35',candidate['architecture'],'h0')
    method.assert_no_fmt_convolution(model)
    assert sum(p.numel() for p in model.parameters())==candidate['parameters']
    return model


def identity(config):
    return dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config_sha256=sha(config),sources={p:sha(p) for p in FILES})


def verify_source(spec,config):
    # Original geometry/labels are read-only, exactly the completed coverage run.
    p35.verify_source(dict(spec,candidate=spec['candidates'][0]),config)
    reports={};root=Path(spec['output']);(root/'centers').mkdir(exist_ok=True)
    for flow in spec['flows']:
        name=flow['name'];reports[name]={}
        for role in ('train','validation','test'):
            folder=Path(spec['source_output'])/'physical'/name/role
            with np.load(folder/'metadata.npz') as z:m={k:z[k] for k in z.files}
            ids,ratio=method.original_center_indices(np.load(folder/'seeds.npy',mmap_mode='r'),m)
            missing=ids<0
            report=dict(total=len(ids),present=int((~missing).sum()),missing=int(missing.sum()),
                missing_positive=int(np.sum(missing&(m['labels']==1))),missing_row_ids=np.flatnonzero(missing).tolist(),
                present_max_distance_over_neighbor_step=float(ratio[~missing].max()),
                missing_min_distance_over_neighbor_step=float(ratio[missing].min()) if missing.any() else None)
            file=root/'centers'/f'{name}_{role}.npz'
            np.savez_compressed(file,center_ids=ids,distance_over_neighbor_step=ratio)
            report['center_index_sha256']=sha(file);reports[name][role]=report
    write(root/'original_center_audit.json',dict(complete=True,identity=identity(config),flows=reports,
        nearest_substitution=False,policy=spec['missing_center_policy']))
    # This lock fixes methods only. Every training run locks its epoch before test loading.
    write(root/'selection.lock.json',dict(complete=True,selected=spec['candidates'],threshold=.5,
        selected_by='user_fixed_two_existing_models_and_original_sampling_center',identity=identity(config)))


class Dataset:
    def __init__(self,spec,role,candidate,device='cuda',limit=None):
        self.device=device;self.role=role;self.parts=[]
        self.evidence_folder=None if limit else Path(spec['output'])/'final'/candidate['id']/f"seed{spec['_active_seed']}"
        if role=='test':
            path=Path(spec['output'])/'final'/candidate['id']/f"seed{spec['_active_seed']}"/'selection.lock.json'
            assert json.loads(path.read_text())['test_loaded'] is False
        labels=[];owners=[];flows=[];rows=[];centers=[]
        for fi,flow in enumerate(spec['flows']):
            name=flow['name'];folder=Path(spec['source_output'])/'physical'/name/role
            with np.load(folder/'metadata.npz') as z:m={k:z[k] for k in z.files}
            with np.load(Path(spec['_root_output'])/'centers'/f'{name}_{role}.npz') as z:anchor=z['center_ids']
            chosen=np.arange(len(anchor))
            if limit:
                # Engineering pilot only; do not claim full-data evaluation.
                chosen=np.concatenate([np.flatnonzero((m['labels']==cls)&(anchor>=0))[:limit//4] for cls in (0,1)])
            elif spec['missing_center_policy']=='present_only_paired':
                chosen=chosen[anchor>=0]
            elif np.any(anchor<0):
                raise ValueError('Original centers missing; a resolved data policy is required. No nearest substitution.')
            self.parts.append((folder,chosen,m['counts'][chosen],anchor[chosen]))
            labels.append(m['labels'][chosen]);owners.append(m['instance'][chosen]);rows.append(chosen)
            flows.append(np.full(len(chosen),fi));centers.append(anchor[chosen])
        self.labels=np.concatenate(labels).astype(np.int64);self.owners=np.concatenate(owners)
        self.flows=np.concatenate(flows);self.rows=np.concatenate(rows);self.anchor=np.concatenate(centers)
        self.targets=torch.tensor(self.labels,device=device)
        if not limit:
            assert len(self.labels)==spec['subset_expected_counts'][role]
            lock=json.loads((Path(spec['_root_output'])/'subset_verification.json').read_text())
            assert lock['complete'] and lock['identity']['config_sha256']==sha(CONFIG if '_config' not in spec else spec['_config'])
            for fi,flow in enumerate(spec['flows']):
                file=Path(spec['_root_output'])/'subsets'/f"{flow['name']}_{role}.npz"
                assert sha(file)==lock['flows'][flow['name']][role]['sha256']
                with np.load(file) as z:
                    assert np.array_equal(self.rows[self.flows==fi],z['row_ids'])
                    assert np.array_equal(self.anchor[self.flows==fi],z['center_ids'])
        self.neighbors=torch.empty((len(self.labels),0),device=device,dtype=torch.long)
        self.clean=None

    def encode(self,candidate,batch=128):
        self.clean=torch.empty((len(self.labels),27 if rotating(candidate) else 1,142),device=self.device)
        selected_neighbors=[];offset=0
        for folder,rows,counts,anchor in self.parts:
            g=np.load(folder/'geometry.npy',mmap_mode='r');s=np.load(folder/'seeds.npy',mmap_mode='r')
            for first in range(0,len(rows),batch):
                take=rows[first:first+batch]
                tokens,neighbors=encode_candidate(torch.tensor(np.array(g[take]),device=self.device),
                    torch.tensor(np.array(s[take]),device=self.device),
                    torch.tensor(counts[first:first+batch],device=self.device,dtype=torch.long),
                    torch.tensor(anchor[first:first+batch],device=self.device,dtype=torch.long),candidate)
                if candidate['base_method']=='p35_h0':
                    a=tokens.cpu().numpy();a[...,:141]=p35.transform_fmt(a[...,:141],True)
                    tokens=torch.from_numpy(a).to(self.device)
                self.clean[offset:offset+len(take)]=tokens;offset+=len(take)
                selected_neighbors.append(neighbors.cpu().numpy())
        assert offset==len(self.labels) and torch.isfinite(self.clean).all()
        self.selected_neighbors=np.concatenate(selected_neighbors)
        if self.evidence_folder is not None:
            write(self.evidence_folder/f'{self.role}_encoding.json',dict(candidate=candidate,
                token_shape=list(self.clean.shape),valid_tokens=int(self.clean[...,-1].sum()),
                original_center_min=int(self.anchor.min()),original_center_max=int(self.anchor.max()),
                original_center_ids_sha256=hashlib.sha256(self.anchor.astype('<i8').tobytes()).hexdigest(),
                neighbors_of_original_center_sha256=hashlib.sha256(self.selected_neighbors.astype('<i8').tobytes()).hexdigest(),
                one_original_center_only=not rotating(candidate),no_missing_original_center=bool(np.all(self.anchor>=0))))


def arm_spec(spec,index):
    arm=copy.deepcopy(spec);arm['_root_output']=spec['output']
    arm['candidate']=spec['candidates'][index];arm['output']=str(Path(spec['output'])/'arms'/arm['candidate']['id'])
    return arm


def fit_normalizer(dataset,candidate,batch=128):
    return p35.fit_normalizer(dataset,candidate,batch) if candidate['base_method']=='p35_h0' else frozen.fit_normalizer(dataset,candidate,batch)


def run_environment(candidate):
    standardize=p35.standardize if candidate['base_method']=='p35_h0' else frozen.standardize
    predict=FunctionType(frozen.predict.__wrapped__.__code__,dict(frozen.__dict__,standardize=standardize),argdefs=frozen.predict.__wrapped__.__defaults__)
    # Keep the original parameterized network, with a strict one-token input gate.
    from types import SimpleNamespace
    constructor=lambda pool,architecture,profile: make_candidate_model(candidate)
    frozen_method=SimpleNamespace(**dict(frozen.method.__dict__,FourierClassifier=constructor))
    return dict(frozen.__dict__,Dataset=Dataset,identity=identity,fit_normalizer=fit_normalizer,method=frozen_method,
                standardize=standardize,predict=torch.no_grad()(predict),save_predictions=save_predictions)


def save_predictions(folder,role,dataset,probability):
    path=folder/f'{role}_predictions.npz'
    np.savez_compressed(path,labels=dataset.labels,probability=probability,flow_index=dataset.flows,
        instance=dataset.owners,row_in_split=dataset.rows,original_center_id=dataset.anchor,threshold=.5)
    return sha(path)


def gpu_check(spec,config):
    frozen.engine.deterministic('cuda');torch.set_num_threads(4)
    assert 'V100' in torch.cuda.get_device_name()
    report={}
    for index,candidate in enumerate(spec['candidates']):
        torch.manual_seed(97011)
        arm=arm_spec(spec,index);d=Dataset(arm,'train',candidate,limit=32);d.encode(candidate)
        env=run_environment(candidate);norm=fit_normalizer(d,candidate)
        model=make_candidate_model(candidate).cuda()
        initial_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
        xx=env['standardize'](d.clean,norm)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);history=[]
        for _ in range(150):
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(xx),d.targets);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            optimizer.step();history.append(float(loss.detach()))
        probability,_=env['predict'](model,d,norm,128);score=frozen.metrics(d.labels,probability)
        assert score['f1']>=.95 and history[-1]<history[0]
        torch.nn.functional.cross_entropy(model(xx.repeat(4,1,1)),d.targets.repeat(4)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        small=Dataset(arm,'train',candidate,limit=32);small.encode(candidate,batch=8)
        assert np.array_equal(d.selected_neighbors,small.selected_neighbors)
        assert torch.allclose(d.clean,small.clean,atol=2e-4,rtol=2e-4)
        # Same material neighborhoods and same original center as frozen multi-center encoder.
        errors=[];offset=0
        for folder,rows,counts,anchor in d.parts:
            g=torch.tensor(np.array(np.load(folder/'geometry.npy',mmap_mode='r')[rows]),device='cuda')
            s=torch.tensor(np.array(np.load(folder/'seeds.npy',mmap_mode='r')[rows]),device='cuda')
            c=torch.tensor(counts,device='cuda');a=torch.tensor(anchor,device='cuda');bi=torch.arange(len(rows),device='cuda')
            if candidate['base_method']=='p35_h0':
                reference=p35.encode_batch(g,s,c)[bi,a].cpu().numpy()
                reference[:,:141]=p35.transform_fmt(reference[:,:141],True)
                reference=torch.tensor(reference,device='cuda')
            else:
                n=frozen.method.neighbor_indices(s,c,'fps6')
                reference=frozen.method.fourier_tokens(frozen.method.resample(g,48,'uniform'),c,n,'p35')[bi,a]
                assert np.array_equal(n[bi,a].cpu().numpy(),d.selected_neighbors[offset:offset+len(rows)])
            actual=d.clean[offset:offset+len(rows)]
            actual=actual[bi,a] if rotating(candidate) else actual[:,0]
            assert torch.allclose(actual,reference,atol=2e-4,rtol=2e-4)
            errors.append(float((actual-reference).abs().max()));offset+=len(rows)
        report[candidate['id']]=dict(samples=32,token_shape=list(d.clean.shape),
            initialization_sha256=initial_hash,
            parameters=sum(p.numel() for p in model.parameters()),pilot_f1=score['f1'],
            initial_loss=history[0],final_loss=history[-1],batch128_backward=True,
            same_original_anchor_features_max_error=max(errors),batch_consistent=True,
            modules=sorted(set(type(m).__name__ for m in model.modules())),network=str(model))
    for name in ('p35_h0','c156'):
        pair=[report[c['id']] for c in spec['candidates'] if c['base_method']==name]
        assert len(set(r['initialization_sha256'] for r in pair))==1
    write(Path(spec['output'])/'paired_gpu_check.json',dict(complete=True,identity=identity(config),
        gpu=torch.cuda.get_device_name(),checks=report,scientific_metric=False))


def train(spec,config,index):
    assert spec['missing_center_policy']=='present_only_paired'
    for filename in ('paired_gpu_check.json','subset_verification.json'):
        evidence=json.loads((Path(spec['output'])/filename).read_text())
        assert evidence['complete'] and evidence['identity']==identity(config)
    spec=dict(spec,_config=config)
    arm=arm_spec(spec,index);arm['_active_seed']=spec['final']['seeds'][0]
    write(Path(arm['output'])/'selection.lock.json',dict(complete=True,selected=arm['candidate'],identity=identity(config)))
    environment=run_environment(arm['candidate'])
    FunctionType(frozen.train.__code__,environment,argdefs=frozen.train.__defaults__)(arm,config,0,'final')


def verify_subset(spec,config):
    """Freeze label-independent row lists without changing any source file."""
    assert spec['missing_center_policy']=='present_only_paired'
    root=Path(spec['output']);(root/'subsets').mkdir(parents=True,exist_ok=True)
    p35.verify_source(dict(spec,output=str(root/'subset_preflight'),candidate=spec['candidates'][0]),config)
    counts={role:0 for role in spec['subset_expected_counts']};reports={}
    for flow in spec['flows']:
        name=flow['name'];reports[name]={}
        for role in counts:
            folder=Path(spec['source_output'])/'physical'/name/role
            with np.load(folder/'metadata.npz') as z:m={k:z[k] for k in z.files}
            ids,ratio=method.original_center_indices(np.load(folder/'seeds.npy',mmap_mode='r'),m)
            with np.load(root/'centers'/f'{name}_{role}.npz') as z:
                assert np.array_equal(ids,z['center_ids'])
            rows=np.flatnonzero(ids>=0);file=root/'subsets'/f'{name}_{role}.npz'
            np.savez_compressed(file,row_ids=rows,center_ids=ids[rows])
            counts[role]+=len(rows)
            reports[name][role]=dict(source=len(ids),retained=len(rows),excluded=int((ids<0).sum()),
                positive=int(m['labels'][rows].sum()),negative=int((m['labels'][rows]==0).sum()),
                sha256=sha(file),max_seed_error_over_spacing=float(ratio[rows].max()),
                positive_per_instance={str(i):int(np.sum((m['instance'][rows]==i)&(m['labels'][rows]==1)))
                    for i in np.unique(m['instance'][m['labels']==1])})
    assert counts==spec['subset_expected_counts']
    write(root/'subset_verification.json',dict(complete=True,identity=identity(config),counts=counts,
        flows=reports,source_modified=False,selection_rule='original_center_id >= 0 only',
        all_four_arms_share_rows=True,source_check_sha256=sha(root/'subset_preflight/source_verification.json')))


def audit(spec,config):
    """Recompute every saved score and verify paired source rows and selection."""
    from sklearn.metrics import average_precision_score,roc_auc_score
    root=Path(spec['output']);subset=json.loads((root/'subset_verification.json').read_text())
    assert subset['complete'] and subset['identity']==identity(config)
    for flow in spec['flows']:
        for role in spec['subset_expected_counts']:
            assert sha(root/'subsets'/f"{flow['name']}_{role}.npz")==subset['flows'][flow['name']][role]['sha256']
            for filename,digest in spec['source_files'][flow['name']][role]['files'].items():
                assert sha(Path(spec['source_output'])/'physical'/flow['name']/role/filename)==digest
    records=[];histories=[];paired_rows={}
    def measure(y,p):
        prediction=p>=.5
        tp=int(np.sum((y==1)&prediction));fp=int(np.sum((y==0)&prediction))
        fn=int(np.sum((y==1)&~prediction));tn=int(np.sum((y==0)&~prediction))
        return dict(samples=len(y),tp=tp,fp=fp,fn=fn,tn=tn,
            f1=2*tp/max(2*tp+fp+fn,1),accuracy=(tp+tn)/len(y),
            precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
            average_precision=float(average_precision_score(y,p)) if y.sum() else None,
            roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None)
    for index,candidate in enumerate(spec['candidates']):
        arm=arm_spec(spec,index);seed=spec['final']['seeds'][0]
        folder=Path(arm['output'])/'final'/candidate['id']/f'seed{seed}'
        result=json.loads((folder/'result.json').read_text())
        assert result['complete'] and result['test_loaded'] and result['identity']==identity(config)
        assert result['candidate']==candidate and result['seed']==seed and result['parameters']==candidate['parameters']
        assert all(h['samples']==spec['subset_expected_counts']['train'] for h in result['history'])
        selected=max(result['history'],key=lambda h:(h['validation_f1'],h['validation_average_precision']))
        lock=json.loads((folder/'selection.lock.json').read_text())
        assert selected['epoch']==result['selected_epoch']==lock['selected_epoch']
        assert lock['test_loaded'] is False and lock['threshold']==.5
        histories.append([h['permutation_sha256'] for h in result['history']]);scores={}
        for role in ('validation','test'):
            file=folder/f'{role}_predictions.npz';assert sha(file)==result['predictions'][role]
            with np.load(file) as z:prediction={k:z[k] for k in z.files}
            assert float(prediction['threshold'])==.5
            y=prediction['labels'];probability=prediction['probability']
            assert len(y)==spec['subset_expected_counts'][role] and np.isfinite(probability).all()
            assert np.all((probability>=0)&(probability<=1))
            key=np.column_stack((prediction['flow_index'],prediction['row_in_split']))
            if role in paired_rows:assert np.array_equal(key,paired_rows[role])
            else:paired_rows[role]=key
            per_flow={};per_instance={};old=np.zeros(len(y),bool)
            for fi,flow in enumerate(spec['flows']):
                name=flow['name'];mask=prediction['flow_index']==fi
                rows=prediction['row_in_split'][mask]
                with np.load(root/'subsets'/f'{name}_{role}.npz') as z:
                    assert np.array_equal(rows,z['row_ids'])
                    assert np.array_equal(prediction['original_center_id'][mask],z['center_ids'])
                with np.load(Path(spec['source_output'])/'physical'/name/role/'metadata.npz') as z:
                    assert np.array_equal(y[mask],z['labels'][rows])
                    assert np.array_equal(prediction['instance'][mask],z['instance'][rows])
                per_flow[name]=measure(y[mask],probability[mask]);per_instance[name]={}
                for instance in np.unique(prediction['instance'][mask&(y==1)]):
                    take=mask&(y==1)&(prediction['instance']==instance)
                    per_instance[name][str(instance)]=dict(positive_samples=int(take.sum()),
                        true_positive=int(np.sum(probability[take]>=.5)),recall=float(np.mean(probability[take]>=.5)))
                if role=='test':old[mask]=rows<spec['old_test_per_flow'][name]
            combined=measure(y,probability)
            expected=result['validation'] if role=='validation' else result['test']['combined']
            for metric in ('f1','accuracy','precision','recall','average_precision'):
                assert abs(combined[metric]-expected[metric])<1e-12,(candidate['id'],role,metric)
            scores[role]=dict(combined=combined,per_flow=per_flow,per_instance=per_instance)
            if role=='test':scores[role].update(original_test_subset=measure(y[old],probability[old]),
                                               added_head_subset=measure(y[~old],probability[~old]))
        fit_tokens=result['normalization']['fit_tokens']
        for role in spec['subset_expected_counts']:
            evidence=json.loads((folder/f'{role}_encoding.json').read_text())
            assert evidence['no_missing_original_center']
            assert evidence['token_shape']==[spec['subset_expected_counts'][role],27 if rotating(candidate) else 1,142]
        train_encoding=json.loads((folder/'train_encoding.json').read_text())
        assert fit_tokens==train_encoding['valid_tokens']
        if not rotating(candidate):assert fit_tokens==spec['subset_expected_counts']['train']
        assert result['normalization']['train_only']
        records.append(dict(candidate=candidate,seed=seed,parameters=result['parameters'],
            selected_epoch=result['selected_epoch'],epochs=result['epochs'],training_seconds=result['training_seconds'],
            preparation_seconds=result['preparation_seconds'],fit_tokens=fit_tokens,scores=scores,
            result_sha256=sha(folder/'result.json')))
    for history in histories[1:]:
        n=min(len(history),len(histories[0]));assert history[:n]==histories[0][:n]
    pairs={}
    for name in ('p35_h0','c156'):
        single=next(r for r in records if r['candidate']['base_method']==name and not rotating(r['candidate']))
        control=next(r for r in records if r['candidate']['base_method']==name and rotating(r['candidate']))
        pairs[name]={role:single['scores'][role]['combined']['f1']-control['scores'][role]['combined']['f1']
                     for role in ('validation','test')}
    assert not list((root/'arms').rglob('*.pt')) and not list((root/'arms').rglob('*.pth'))
    write(root/'summary.json',dict(complete=True,identity=identity(config),counts=spec['subset_expected_counts'],
        methods=records,single_minus_rotating_f1=pairs,paired_rows_and_epoch_permutations_verified=True,
        single_seed_only=True,threshold=.5,no_weight_files=True))
    print(json.dumps(dict(complete=True,paired_differences=pairs)),flush=True)


def submit_formal(spec,config):
    assert spec['missing_center_policy']=='present_only_paired' and len(spec['candidates'])==4
    root=Path(spec['output']);(root/'logs').mkdir(parents=True,exist_ok=True)
    assert not (root/'submission_formal.json').exists()
    jobs={}
    for phase,dependency,array,gpu,wall,memory in [
        ('verify-subset',None,None,False,'00:15:00','16G'),
        ('gpu-check','verify-subset',None,True,'00:10:00','32G'),
        ('train','gpu-check','0-3%4',True,'06:00:00','64G'),
        ('audit','train',None,False,'00:15:00','16G')]:
        command=['sbatch','--parsable','--propagate=NONE','--job-name=OriginalCenter_'+phase,
            '--cpus-per-task=4','--mem='+memory,'--time='+wall,
            '--output='+str(root/'logs/%x_%A_%a.out'),'--error='+str(root/'logs/%x_%A_%a.err')]
        if dependency:command+=['--dependency=afterok:'+jobs[dependency]]
        if array:command+=['--array='+array]
        if gpu:command+=['--gres=gpu:1','--constraint=v100','--exclude=gpu213-18']
        command+=['ibex_bash/task4c_original_center_1p1.sh',phase,config]
        jobs[phase]=subprocess.check_output(command,text=True).strip().split(';')[0]
        frozen.append_locked(root/'submissions.jsonl',json.dumps(dict(phase=phase,job=jobs[phase],command=command))+'\n')
        write(root/'submission_formal.json',dict(identity=identity(config),jobs=jobs,formal_training_submitted='train' in jobs))
    print(json.dumps(jobs),flush=True)


def diagnose_centers(spec,config,index):
    """Read-only reintegration check: never changes the scientific dataset."""
    from FMT_Utils.Task4C_PaperBundles_3_1 import load_flow
    from experiments.Task4C_PhysicalLength_4_14 import trace_lines
    physical=json.loads(Path('config/mainExp_Task4C_GTHeadCoverage_1.1.json').read_text())
    flow=physical['flows'][index];name=flow['name'];root=Path(spec['output'])
    axes,_,_,_,grid,_=load_flow(Path(physical['input_root'])/flow['flow'],flow['lambda2_threshold'])
    reports={}
    for role in ('train','validation','test'):
        folder=Path(spec['source_output'])/'physical'/name/role
        with np.load(folder/'metadata.npz') as z:m={k:z[k] for k in z.files}
        with np.load(root/'centers'/f'{name}_{role}.npz') as z:ids=np.flatnonzero(z['center_ids']<0)
        recovered=np.zeros((len(ids),32,3),np.float32);valid=np.zeros(len(ids),bool);statistics=[]
        for scale_id in np.unique(m['scale_id'][ids]):
            selected=np.flatnonzero(m['scale_id'][ids]==scale_id);rows=ids[selected]
            scale=dict(ds=float(m['ds'][rows[0]]),maxiteration=int(m['maxiteration'][rows[0]]))
            geometry,accepted,arcs,steps,stats=trace_lines(grid,m['center'][rows],scale,physical)
            recovered[selected]=geometry;valid[selected]=accepted
            statistics.append(dict(scale_id=int(scale_id),scale=scale,stats=stats))
        file=root/'centers'/f'{name}_{role}_reintegration.npz'
        np.savez_compressed(file,row_ids=ids,physical_centerline=recovered,passes_old_cleaning=valid)
        reports[role]=dict(missing=len(ids),passes_old_cleaning=int(valid.sum()),fails_old_cleaning=int((~valid).sum()),
            failed_row_ids=ids[~valid].tolist(),statistics=statistics,sha256=sha(file))
    write(root/f'center_reintegration_{name}.json',dict(complete=True,identity=identity(config),flow=name,
        read_only_diagnosis=True,source_data_modified=False,reports=reports))


def runtime(spec,config,phase,state,code):
    frozen.append_locked(Path(spec['output'])/'runtime.jsonl',json.dumps(dict(identity=identity(config),
        phase=phase,state=state,exit_code=code,hostname=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),
        array_index=os.environ.get('SLURM_ARRAY_TASK_ID'),at_utc=datetime.now(timezone.utc).isoformat()))+'\n')


def submit_preflight(spec,config):
    root=Path(spec['output']);(root/'logs').mkdir(parents=True,exist_ok=True)
    assert not (root/'submission_preflight.json').exists()
    jobs={}
    for phase,dependency,array,gpu in [('verify-source',None,None,False),('gpu-check','verify-source',None,True),
                                      ('diagnose-centers','verify-source','0-1',False)]:
        command=['sbatch','--parsable','--propagate=NONE','--job-name=OriginalCenter_'+phase,
            '--cpus-per-task=4','--mem='+('48G' if phase=='diagnose-centers' else '16G'),'--time=00:30:00',
            '--output='+str(root/'logs/%x_%A_%a.out'),'--error='+str(root/'logs/%x_%A_%a.err')]
        if dependency:command+=['--dependency=afterok:'+jobs[dependency]]
        if array:command+=['--array='+array]
        if gpu:command+=['--gres=gpu:1','--constraint=v100','--exclude=gpu213-18']
        command+=['ibex_bash/task4c_original_center_1p1.sh',phase,config]
        jobs[phase]=subprocess.check_output(command,text=True).strip().split(';')[0]
        frozen.append_locked(root/'submissions.jsonl',json.dumps(dict(phase=phase,job=jobs[phase],command=command))+'\n')
        write(root/'submission_preflight.json',dict(identity=identity(config),jobs=jobs,formal_training_submitted=False))
    print(json.dumps(jobs),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['verify-source','verify-subset','gpu-check','diagnose-centers','train','audit','submit-formal','submit-preflight','runtime'])
    p.add_argument('--config',default=CONFIG);p.add_argument('--index',type=int,default=0)
    p.add_argument('--runtime-phase');p.add_argument('--state');p.add_argument('--exit-code',type=int)
    a=p.parse_args();spec=json.loads(Path(a.config).read_text())
    if a.phase=='verify-source':verify_source(spec,a.config)
    elif a.phase=='gpu-check':gpu_check(spec,a.config)
    elif a.phase=='diagnose-centers':diagnose_centers(spec,a.config,a.index)
    elif a.phase=='train':train(spec,a.config,a.index)
    elif a.phase=='verify-subset':verify_subset(spec,a.config)
    elif a.phase=='audit':audit(spec,a.config)
    elif a.phase=='submit-formal':submit_formal(spec,a.config)
    elif a.phase=='submit-preflight':submit_preflight(spec,a.config)
    else:runtime(spec,a.config,a.runtime_phase,a.state,a.exit_code)


if __name__=='__main__':main()
