"""Frozen Task4-c data/trainer, shrinking the BiLSTM plus MLP to 56,753 parameters."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from types import FunctionType
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from FMT_Utils.Task4C_BiLSTMCapacity_1_1 import make_model,parameter_counts,PARAMETERS
from experiments import Task4C_GeometricBaselines_1_1 as geometry
from experiments import Task4C_BottomDensity_1_2 as source
from experiments import Task4C_InstanceCoverage_9_15_v2 as engine

CONFIG='config/Ablation_Task4C_BiLSTMCapacity_1.1.json'
write,sha,metrics,deterministic=engine.write,engine.sha,engine.metrics,engine.deterministic


def load_spec(config):
    definition=json.loads(Path(config).read_text())
    assert set(definition)=={'version','user_version','execution_revision','output','base_config','base_config_sha256',
        'source_output','source_scientific_commit','dataset_policy','methods','bilstm'}
    assert sha(definition['base_config'])==definition['base_config_sha256']
    spec=source.load_spec(definition['base_config']);spec.update(definition)
    assert spec['methods']==['baseline2'] and spec['dataset_policy']=='reuse_all_source_geometry_seeds_metadata_without_sampling'
    b=spec['bilstm'];assert b['layers']==1 and b['input_channels']==3 and b['hidden_per_direction']==71
    assert b['mlp']==[284,47,2] and b['total_parameters']==PARAMETERS<=b['user_corrected_total_parameter_target']
    assert b['recurrent_parameters']==43168 and b['classifier_parameters']==13585
    assert b['dropout']==spec['training']['dropout']==.15
    return spec


def identity(config):
    result=geometry.identity(config)
    for file in ('FMT_Utils/Task4C_BiLSTMCapacity_1_1.py','experiments/Task4C_BiLSTMCapacity_1_1.py',
                 'ibex_bash/task4c_bilstm_capacity_1p1.sh'):
        result['sources'][file]=sha(file)
    return result


def run_engine(name,*args):
    fn=getattr(engine,name)
    return FunctionType(fn.__code__,dict(engine.__dict__,identity=identity),argdefs=fn.__defaults__)(*args)


def reuse(spec,config):
    fn=geometry.reuse
    FunctionType(fn.__code__,dict(geometry.__dict__,identity=identity),argdefs=fn.__defaults__)(spec,config)


def train(spec,config,index):
    # Reuse the original training function's exact code object, changing only the model factory and identity.
    fn=geometry.train
    FunctionType(fn.__code__,dict(geometry.__dict__,identity=identity,make_model=make_model),argdefs=fn.__defaults__)(spec,config,index)


def preflight(spec,config,device):
    deterministic(device);torch.manual_seed(91731)
    model=make_model(dropout=.15).to(device);counts=torch.tensor([10,19,27,15],device=device)
    g=torch.randn((4,27,32,3),device=device)*.15
    valid=torch.arange(27,device=device)[None]<counts[:,None]
    changed=g.clone();changed[~valid]=999
    reordered=g.clone()
    for b,n in enumerate(counts.tolist()):reordered[b,:n]=g[b,:n].flip(0)
    model.eval()
    with torch.no_grad():
        expected=model((g,counts))
        assert torch.equal(expected,model((changed,counts)))
        assert torch.allclose(expected,model((reordered,counts)),atol=2e-6,rtol=2e-6)
        assert not torch.allclose(expected,model((g.flip(2),counts)),atol=1e-6,rtol=1e-6)
        assert torch.allclose(expected[:1],model((g[:1],counts[:1])),atol=2e-6,rtol=2e-6)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    for _ in range(3):
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model((g,counts)),torch.tensor([0,1,0,1],device=device))
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    result=dict(complete=True,identity=identity(config),device=device,parameters=parameter_counts(model),
        padding_excluded=True,line_permutation_invariance=True,within_line_order_used=True,finite_forward_backward=True,
        trainer='exact_original_GeometricBaselines_1_1_train_code_with_model_factory_override')
    write(Path(spec['output'])/f'preflight_{device}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='identity'}),flush=True)


def pilot(spec,config):
    deterministic('cuda');torch.manual_seed(91732);root=Path(spec['output']);gg=[];cc=[];yy=[]
    for flow in spec['flows']:
        folder=root/'physical'/flow['name']/'train';g=np.load(folder/'geometry.npy',mmap_mode='r')
        with np.load(folder/'metadata.npz') as z:
            rows=np.concatenate([np.flatnonzero(z['labels']==cls)[:8] for cls in (0,1)])
            gg.append(np.array(g[rows]));cc.append(z['counts'][rows]);yy.append(z['labels'][rows])
    x=(torch.as_tensor(np.concatenate(gg),device='cuda'),torch.as_tensor(np.concatenate(cc).astype(np.int64),device='cuda'))
    y=np.concatenate(yy);target=torch.as_tensor(y,device='cuda',dtype=torch.long)
    model=make_model(dropout=.15).cuda().eval()
    with torch.no_grad():initial=float(F.cross_entropy(model(x),target))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);start=time.perf_counter()
    for _ in range(100):
        model.train();optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x),target)
        assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    model.eval()
    with torch.no_grad():
        logits=model(x);final=float(F.cross_entropy(logits,target));prob=logits.softmax(-1)[:,1].cpu().numpy()
    assert final<initial,(initial,final)
    fit_seconds=time.perf_counter()-start;big=(x[0].repeat(4,1,1,1),x[1].repeat(4))
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(3):
        model.train();optimizer.zero_grad(set_to_none=True);F.cross_entropy(model(big),target.repeat(4)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
    torch.cuda.synchronize();batch_seconds=(time.perf_counter()-start)/3
    result=dict(complete=True,identity=identity(config),train_only=True,samples=32,parameters=parameter_counts(model),
        initial_loss=initial,final_loss=final,training_metrics=metrics(y,prob),fit100_seconds=fit_seconds,
        batch128_forward_backward=True,batch128_step_seconds=batch_seconds,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    write(root/'pilot.json',result);print(json.dumps({k:v for k,v in result.items() if k!='identity'}),flush=True)


def merge(spec,config):
    fn=geometry.reuse_source.merge
    FunctionType(fn.__code__,dict(geometry.reuse_source.__dict__,identity=identity,run_engine=run_engine),argdefs=fn.__defaults__)(spec,config)
    file=Path(spec['output'])/'summary.json';summary=json.loads(file.read_text())
    assert all(row['parameters']==PARAMETERS for row in summary['methods'])
    file=Path(spec['output'])/'viewer_package/manifest.json';manifest=json.loads(file.read_text())
    manifest['evidence_note']='BiLSTM hidden71 plus MLP284/47/2; total56753 learned parameters on frozen fivefold data; seed96611.'
    for model in manifest['models']:model['label']='BiLSTM + MLP (56,753 parameters)'
    write(file,manifest)


def submit(spec,config):
    root=Path(spec['output']).resolve();assert shutil.disk_usage(Path.cwd()).free>5*2**30
    root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir();write(root/'config.frozen.json',spec);dependency=None
    phases=[('preflight',None,True,'00:20:00'),('reuse',None,False,'00:30:00'),('pilot',None,True,'00:30:00'),
        ('train','0-2%3',True,'24:00:00'),('merge',None,False,'01:00:00')]
    for phase,array,gpu,limit in phases:
        command=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=32G','--time='+limit,
            '--job-name=t4c-bilstm57k-'+phase,f'--output={root}/logs/{phase}_%A_%a.out',f'--error={root}/logs/{phase}_%A_%a.err']
        if array:command+=['--array='+array]
        if gpu:command+=['--gres=gpu:1','--constraint=v100']
        if dependency:command+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
        command+=['ibex_bash/task4c_bilstm_capacity_1p1.sh',phase,config]
        job=subprocess.check_output(command,text=True).strip().split(';')[0]
        row=dict(job_id=job,phase=phase,submitted_at_utc=datetime.now(timezone.utc).isoformat(),command=command,
            dependency=dependency,expected_device='V100' if gpu else 'CPU',identity=identity(config))
        engine.append_locked(root/'submissions.jsonl',json.dumps(row)+'\n');print(json.dumps(row),flush=True);dependency=job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','reuse','pilot','train','merge','submit','runtime'))
    parser.add_argument('--config',default=CONFIG);parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--device',default='cuda');parser.add_argument('--runtime-phase')
    parser.add_argument('--state');parser.add_argument('--exit-code',type=int)
    args=parser.parse_args();spec=load_spec(args.config)
    if args.phase=='preflight':preflight(spec,args.config,args.device)
    elif args.phase=='train':train(spec,args.config,args.index)
    elif args.phase=='runtime':run_engine('runtime',spec,args.config,args.runtime_phase,args.state,args.exit_code)
    else:globals()[args.phase](spec,args.config)


if __name__=='__main__':main()
