"""Single-seed, five-model ball6/16 comparison on frozen Task4-c v3."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from datetime import datetime, timezone
from types import FunctionType
import numpy as np
import torch
from experiments import Task4C_BallQuery_2_5 as fmt_old
from experiments import Task4C_BallQueryBaselines_2_5 as base_old
from FMT_Utils import Task4C_V3Training_3_1 as data
from experiments.Prepare_Task4C_BallQuery_3_1 import write

CONFIG='config/Ablation_Task4C_BallQuery_3.1.json'
BASE_CONFIGS={k:f'config/Ablation_Task4C_BallQueryBaselines{k}_3.1.json' for k in (6,16)}
FILES=tuple(dict.fromkeys(base_old.FILES+('FMT_Utils/Task4C_V3Training_3_1.py',
    'experiments/Task4C_V3Training_3_1.py','experiments/Submit_Task4C_V3Training_3_1.py',
    'tests/test_task4c_v3_training_3_1.py','ibex_bash/task4c_v3_training_3p1.sh',
    'FMT_Utils/Task4C_BallQuery_3_1.py','experiments/Prepare_Task4C_BallQuery_3_1.py',
    'config/Verify_Task4C_BallQuery_3.1.json',
    'docs/Task4C_v3_training_protocol_3.1.md',CONFIG)+tuple(BASE_CONFIGS.values())))
sha=fmt_old.baseline.sha


def identity(config):
    return dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                config_sha256=sha(config),sources={p:sha(p) for p in FILES})


def fmt_spec():
    s=json.loads(Path(CONFIG).read_text())
    assert s['final']['seeds']==[96611] and len(s['candidates'])==4
    return s


def baseline_spec(index):
    k=(6,16)[index//3];family=base_old.FAMILIES[index%3];config=BASE_CONFIGS[k]
    s=base_old.load_spec(config,family);definition=json.loads(Path(config).read_text())
    s['neighbor_output']=definition['neighbor_output']
    return s,config


def check_report(path,config):
    r=json.loads(Path(path).read_text());assert r['complete'] and r['identity']==identity(config)
    return r


def verify_source():
    s=fmt_spec();source=Path(s['source_output'])
    assert sha(source/'data_audit.json')==s['source_audit_sha256']
    audit=json.loads((source/'data_audit.json').read_text());assert audit['complete']
    counts={r:0 for r in ('train','validation','test')};tables={}
    for fi,f in enumerate(s['flows']):
        name=f['name'];folder=source/'physical'/name
        for file,digest in audit['frozen_files'][name].items(): assert sha(folder/file)==digest
        neighbor=Path(s['neighbor_output'])/name
        m=json.loads((neighbor/'manifest.json').read_text())
        assert m['complete'] and m['source_audit_sha256']==s['source_audit_sha256']
        assert m['h']==f['h_value'] and m['radius_h']==3 and m['independent_centers']>=30
        for file,digest in m['files'].items(): assert sha(neighbor/file)==digest
        with np.load(folder/'metadata.npz') as z:
            masks=data.v2.split_masks({'fold':z['fold']},fi,s)
        for role,mask in masks.items(): counts[role]+=int(mask.sum())*3
        tables[name]=dict(manifest_sha256=sha(neighbor/'manifest.json'),samples=m['samples'],statistics=m['initial_count'])
    assert counts==s['expected_counts']
    write(Path(s['output'])/'source_verification.json',dict(complete=True,identity=identity(CONFIG),counts=counts,
        neighbor_tables=tables,source_audit_sha256=s['source_audit_sha256']))
    print(json.dumps(dict(source='PASS',counts=counts)),flush=True)


def fit_normalizer(dataset,candidate,batch=128):
    old=fmt_old.baseline
    if candidate['base_method']=='p35_h0' and isinstance(dataset.clean,data.HostTokens):
        # Original NumPy float64 population mean/std; source is a CPU memory map.
        values=dataset.clean.array[:,0,:141]
        mean=values.mean(0,dtype=np.float64);std=values.std(0,dtype=np.float64);std[std<1e-8]=1
        return torch.tensor(mean,device=dataset.device),torch.tensor(std,device=dataset.device),len(values)
    return old.previous.fit_normalizer(dataset,candidate,batch)


def fmt_environment(candidate):
    return dict(fmt_old.baseline.environment(candidate),Dataset=data.Dataset,identity=identity,fit_normalizer=fit_normalizer)


def baseline_environment(spec):
    module=base_old.frozen.MODULES[spec['family']]
    predict=FunctionType(base_old.frozen.engine.predict.__code__,dict(base_old.frozen.engine.__dict__,get_batch=data.baseline_batch))
    return module,dict(module.__dict__,identity=identity,load_parts=data.baseline_parts,
        get_batch=data.baseline_batch,predict=predict,append_locked=base_old.frozen.append_line)


def make_baseline(spec):
    if spec['family']=='conv': return base_old.frozen.engine.Conv915(spec['training']['dropout'])
    if spec['family']=='bilstm': return base_old.frozen.geometry.make_model('baseline2',spec['training']['dropout'])
    return base_old.frozen.pointnet.make_model(spec['training']['dropout'])


def preflight(index):
    spec=fmt_spec();check_report(Path(spec['output'])/'source_verification.json',CONFIG)
    fmt_old.baseline.deterministic('cuda');torch.manual_seed(96611)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    if index<4:
        c=spec['candidates'][index];arm=fmt_old.baseline.arm_spec(spec,index)
        d=data.Dataset(arm,'train',c,limit=32);d.encode(c)
        small=data.Dataset(arm,'train',c,limit=32);small.encode(c,batch=8)
        np.testing.assert_array_equal(d.selected_neighbors,small.selected_neighbors)
        torch.testing.assert_close(d.clean,small.clean,atol=2e-4,rtol=2e-4)
        env=fmt_environment(c);norm=fit_normalizer(d,c);x=env['standardize'](d.clean,norm);y=d.targets
        model=fmt_old.baseline.method.make_model(c['base_method']).cuda();parameters=c['parameters']
        assert not any(isinstance(m,(torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d)) for m in model.modules())
        # Exercise host-backed random indexing using the same encoded values.
        host=data.HostTokens(d.clean.cpu().numpy(),'cuda');take=torch.tensor([7,1,5],device='cuda')
        torch.testing.assert_close(host[take],d.clean[take],atol=0,rtol=0)
        id_=c['id'];penalty=lambda:0.
    else:
        s,config=baseline_spec(index-4);dataset=data.baseline_parts(s,'train',s['methods'][0],limit=32)
        parts,ids,y,_,_=dataset;x=data.baseline_batch(parts,ids,s['methods'][0],None)
        y=torch.tensor(y,device='cuda');model=make_baseline(s).cuda();id_=f"{s['family']}_ball{s['neighbor_count']}"
        parameters={'conv':72192,'bilstm':76786,'pointnet':76749}[s['family']]
        # Old export normalization/padding, compared on real rows before any training.
        from experiments.Task4C_BallQueryBaselines_2_5 import bundle_function
        for fi,p in enumerate(parts):
            source=p['source'];samples=p['samples'][:5];lengths=np.arange(len(samples))%3
            # Minimal metadata for the frozen export helper.
            folder=Path(s['source_output'])/'physical'/s['flows'][fi]['name']
            with np.load(folder/'metadata.npz') as z:
                m={key:z[key] for key in ('label','component','instance','half_arc_lengths','half_step_counts','local_grid_scale')}
            g,_,_=bundle_function(s['neighbor_count'])(source.curves,source.seeds,m,source.neighbors,samples,lengths,
                [70,100,130] if fi==0 else [140,200,260],.001 if fi==0 else .025)
            local=np.column_stack((np.full(len(samples),fi),np.arange(len(samples))*3+lengths))
            actual,_=data.baseline_geometry(parts,local)
            np.testing.assert_array_equal(actual,g)
        if s['family']=='conv':
            g,cnt=data.baseline_geometry(parts,ids[:16]);e=base_old.frozen.engine
            off,ii,v=e.sparse_voxels(torch.tensor(g,device='cuda'),torch.tensor(cnt,device='cuda'),16)
            recovered=e.dense_sparse_batch([dict(offsets=off,indices=ii,values=v)],[(0,i) for i in range(len(g))],16,'cuda')
            torch.testing.assert_close(x[:16],recovered,atol=0,rtol=0)
        penalty=(lambda:s['pointnet']['feature_transform_regularization_weight']*model.orthogonality_penalty()) if s['family']=='pointnet' else (lambda:0.)
    assert sum(p.numel() for p in model.parameters())==parameters
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    model.eval()
    with torch.no_grad(): initial=float(torch.nn.functional.cross_entropy(model(x),y))
    for _ in range(30):
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(x),y)+penalty();assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    model.eval()
    with torch.no_grad(): final=float(torch.nn.functional.cross_entropy(model(x),y))
    assert final<initial
    # Formal batch 128 backward, independent of the balanced pilot row count.
    take=torch.arange(128,device='cuda')%len(y)
    full=tuple(t[take] for t in x) if isinstance(x,tuple) else x[take]
    model.train();optimizer.zero_grad(set_to_none=True)
    (torch.nn.functional.cross_entropy(model(full),y[take])+penalty()).backward();torch.cuda.synchronize()
    write(Path(spec['output'])/'checks'/f'{index}.json',dict(complete=True,identity=identity(CONFIG),engineering_only=True,
        method=id_,parameters=parameters,initial_loss=initial,final_loss=final,batch128_backward=True,
        seconds=time.perf_counter()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name()))


def train(index):
    spec=fmt_spec();root=Path(spec['output'])
    check_report(root/'source_verification.json',CONFIG);check_report(root/'checks'/f'{index}.json',CONFIG)
    if index<4:
        arm=fmt_old.baseline.arm_spec(spec,index)
        write(Path(arm['output'])/'selection.lock.json',dict(complete=True,selected=arm['candidate'],seed=96611,identity=identity(CONFIG)))
        f=fmt_old.baseline.frozen.train
        FunctionType(f.__code__,fmt_environment(arm['candidate']),argdefs=f.__defaults__)(arm,CONFIG,0,'final')
    else:
        s,config=baseline_spec(index-4);module,env=baseline_environment(s)
        FunctionType(module.train.__code__,env,argdefs=module.train.__defaults__)(s,config,0)


def result_location(spec,index):
    if index<4:
        arm=fmt_old.baseline.arm_spec(spec,index);c=arm['candidate']
        return Path(arm['output'])/'final'/c['id']/'seed96611',CONFIG,c['id']
    s,config=baseline_spec(index-4)
    return Path(s['output'])/'runs'/s['methods'][0]/'seed96611',config,f"{s['family']}_ball{s['neighbor_count']}"


def audit_results():
    from sklearn.metrics import f1_score
    spec=fmt_spec();records={}
    truth={}
    for fi,f in enumerate(spec['flows']):
        with np.load(Path(spec['source_output'])/'physical'/f['name']/'metadata.npz') as z:
            meta={key:z[key] for key in ('label','fold','instance')}
        masks=data.v2.split_masks(meta,fi,spec)
        truth[fi]=(meta,{role:np.repeat(np.flatnonzero(mask),3) for role,mask in masks.items()})
    for index in range(10):
        folder,config,name=result_location(spec,index);r=check_report(folder/'result.json',config)
        assert r['seed']==96611
        expected_parameters=spec['candidates'][index]['parameters'] if index<4 else (72192,76786,76749)[(index-4)%3]
        assert r['parameters']==expected_parameters and r['epochs']==len(r['history'])
        assert all(row['samples']==spec['expected_counts']['train'] for row in r['history'])
        selected=max(r['history'],key=lambda h:(h['validation_f1'],h['validation_average_precision']))['epoch']
        assert selected==r['selected_epoch']
        lock=json.loads((folder/'selection.lock.json').read_text());assert lock['test_loaded'] is False and lock['selected_epoch']==selected
        scores={}
        for role in ('validation','test'):
            file=folder/f'{role}_predictions.npz';assert sha(file)==r['predictions'][role]
            with np.load(file) as z: p={key:z[key] for key in z.files}
            assert len(p['labels'])==spec['expected_counts'][role] and float(p['threshold'])==.5
            assert np.isfinite(p['probability']).all() and ((p['probability']>=0)&(p['probability']<=1)).all()
            for fi in range(2):
                take=p['flow_index']==fi;meta,ids=truth[fi]
                sample_ids=p['sample_id'][take] if index<4 else ids[role][p['row_in_split'][take]]
                np.testing.assert_array_equal(sample_ids,ids[role])
                np.testing.assert_array_equal(p['labels'][take],meta['label'][sample_ids])
                np.testing.assert_array_equal(p['instance'][take],meta['instance'][sample_ids])
            actual=float(f1_score(p['labels'],p['probability']>=.5))
            expected=(r['validation'] if role=='validation' else r['test']['combined']) if index<4 else r['metrics'][role]['combined']
            assert abs(actual-expected['f1'])<1e-12
            scores[role]=dict(combined=base_old.frozen.engine.metrics(p['labels'],p['probability']),
                per_flow={f['name']:base_old.frozen.engine.metrics(p['labels'][p['flow_index']==fi],p['probability'][p['flow_index']==fi]) for fi,f in enumerate(spec['flows'])})
        records[name]=dict(seed=96611,parameters=r['parameters'],selected_epoch=selected,epochs=r['epochs'],
            training_seconds=r['training_seconds'],scores=scores,result_sha256=sha(folder/'result.json'))
    write(Path(spec['output'])/'summary.json',dict(complete=True,identity=identity(CONFIG),source_audit_sha256=spec['source_audit_sha256'],methods=records))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['verify-source','preflight','train','audit-results','runtime'])
    p.add_argument('--index',type=int,default=0);p.add_argument('--state');p.add_argument('--runtime-phase');p.add_argument('--exit-code',type=int)
    a=p.parse_args()
    if a.phase=='runtime':
        root=Path(fmt_spec()['output']);root.mkdir(parents=True,exist_ok=True)
        base_old.frozen.append_line(root/'runtime.jsonl',json.dumps(dict(identity=identity(CONFIG),phase=a.runtime_phase,state=a.state,
            index=a.index,exit_code=a.exit_code,host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'),at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
    elif a.phase=='verify-source':verify_source()
    elif a.phase=='preflight':preflight(a.index)
    elif a.phase=='train':train(a.index)
    else:audit_results()


if __name__=='__main__':main()
