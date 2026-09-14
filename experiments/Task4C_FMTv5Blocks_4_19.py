"""Fast paired block ablations; validation only, with fixed model capacity."""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from experiments import Task4C_RetainDirection_4_16 as parent
from experiments import Task4C_NearbySampling_4_13 as frozen
from FMT_Utils.Task4C_RetainDirection_4_16 import make_model
from FMT_Utils.Task4C_Multiscale_4_1 import transform_fmt

pipeline = parent.pipeline
identity = parent.identity
CONFIG = 'config/Ablation_Task4C_FMTv5Blocks_4.19.json'
SPLITS = ('train', 'validation')


def load_spec():
    definition = json.loads(Path(CONFIG).read_text())
    assert pipeline.sha(definition['base_config']) == definition['base_config_sha256']
    assert pipeline.sha('experiments/Task4C_NearbySampling_4_13.py') == definition['frozen_trainer_sha256']
    spec = parent.load_spec(definition['base_config']); spec.update(definition)
    spec['training']['seeds'] = [spec['seed']]
    assert list(spec['blocks'].values()) == [[0,23],[23,161],[161,233],[233,323],[323,398]]
    return spec


def reference(spec):
    p = Path(spec['source_output'])/'runs'/f"regularized_fmt_mlp_seed{spec['seed']}"/'result.json'
    assert pipeline.sha(p) == spec['reference_result_sha256']
    r = json.loads(p.read_text())
    assert r['encoder'] == 'fmt_v5_4_14' and r['identity']['commit'] == spec['source_commit']
    assert r['device'] == spec['required_device'] and r['parameters'] == 109634
    assert r['identity']['config_sha256'] == spec['base_config_sha256']
    return p, r


class MaskedFMT(nn.Module):
    def __init__(self, block=None, dropout=.15):
        super().__init__()
        self.network = make_model('fmt_v5_4_14', dropout)
        mask = torch.ones(399)
        if block is not None:
            start, stop = block
            assert 0 <= start < stop <= 398
            mask[start:stop] = 0
        self.register_buffer('feature_mask', mask)

    def forward(self, tokens):
        return self.network(tokens*self.feature_mask)


def load_data(spec, split, manifest):
    if split not in SPLITS:
        raise ValueError('This diagnostic does not load test features')
    return pipeline.load_data(spec, split, 'fmt_mlp', manifest,
                              dict(architecture='learned_pool_wider_conv_4.3'))


def metric(y, p):
    y = np.asarray(y).astype(bool); p = np.asarray(p)
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    pred = p >= .5
    tp = int(np.sum(y & pred)); fp = int(np.sum(~y & pred)); fn = int(np.sum(y & ~pred))
    return dict(f1=2*tp/max(2*tp+fp+fn,1), average_precision=float(average_precision_score(y,p)))


def preflight(spec):
    root = Path(spec['output']); source = Path(spec['source_output'])
    path, ref = reference(spec)
    original = json.loads((source/'encoding.json').read_text())
    assert original['identity']['commit'] == spec['source_commit']
    assert original['identity']['config_sha256'] == spec['base_config_sha256']
    for name, h in original['identity']['source_sha256'].items():
        assert pipeline.sha(name) == h, name
    manifest = dict(identity=identity(CONFIG), splits={}, source_encoding_sha256=pipeline.sha(source/'encoding.json'))
    for flow in spec['flows']:
        for split in SPLITS:
            key = flow['name']+'/'+split; dest = root/key; dest.mkdir(parents=True,exist_ok=False)
            files = {}
            for name in ('fmt.npy','metadata.npz'):
                src = source/key/name; assert pipeline.sha(src) == original['splits'][key]['files'][name]
                (dest/name).symlink_to(src.resolve()); files[name] = pipeline.sha(src)
            manifest['splits'][key] = dict(files=files)
    pipeline.write(root/'encoding.json',manifest)
    # The reference contains historical test results; only train/validation fields
    # and arrays are copied or used by this diagnostic.
    kept = {k:ref[k] for k in ('training','validation','normalization','selected_epoch','parameters','seed','device')}
    kept.update(result_sha256=pipeline.sha(path), source_commit=spec['source_commit'])
    pipeline.write(root/'full_reference.json',kept)
    with np.load(path.parent/'predictions.npz') as z:
        np.savez_compressed(root/'full_reference_predictions.npz',
            **{k:z[k] for k in z.files if k.startswith(('train_','validation_'))})
    assert torch.cuda.get_device_name(0) == spec['required_device']
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    samples = []
    for flow in spec['flows']:
        src = source/flow['name']/'train'
        with np.load(src/'metadata.npz') as z:
            ids = np.concatenate([np.flatnonzero(z['labels']==c)[:8] for c in (0,1)])
        samples.append(np.array(np.load(src/'fmt.npy',mmap_mode='r')[ids]))
    values = np.concatenate(samples); mask = values[...,-1:]
    x = transform_fmt(values[...,:-1],True)
    x = np.clip(((x-np.array(ref['normalization']['mean']))/np.array(ref['normalization']['std'])).astype(np.float32),-8.,8.)
    x = torch.as_tensor(np.concatenate((x*mask,mask),-1),device='cuda')
    torch.manual_seed(spec['seed']); original_model = make_model('fmt_v5_4_14',.15).cuda().eval()
    checks = []
    for name, block in [('full',None),*spec['blocks'].items()]:
        torch.manual_seed(spec['seed']); model = MaskedFMT(block).cuda().eval()
        assert all(torch.equal(v,model.network.state_dict()[k]) for k,v in original_model.state_dict().items())
        assert sum(p.numel() for p in model.parameters()) == 109634 and model.feature_mask[-1] == 1
        if block is None:
            assert torch.equal(model(x), original_model(x))
        else:
            start,stop = block; changed=x.clone();changed[...,start:stop] += 17
            assert torch.equal(model(x),model(changed))
            model(x).square().mean().backward()
            assert torch.all(model.network.line[0].weight.grad[:,start:stop] == 0)
            assert torch.isfinite(model.network.line[0].weight.grad).all()
        checks.append(dict(block=name,initial_trainable_weights_equal=True,parameters=109634,mask_preserved=True))
    # Keep the optimization/early-stopping loop literally equivalent to 4.14.
    def loop(fn):
        tree=ast.parse(inspect.getsource(fn))
        return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
    assert ast.dump(loop(train),include_attributes=False) == ast.dump(loop(frozen.train),include_attributes=False)
    pipeline.write(root/'preflight.json',dict(status='PASS',identity=identity(CONFIG),checks=checks,
        cached_inputs_unchanged=True,training_loop_ast_matches_frozen=True,test_features_loaded=False))


def train(spec, index):
    block = list(spec['blocks'])[index]; seed=spec['seed']; candidate=spec['candidates'][0]; method='fmt_mlp'
    out=Path(spec['output'])/'runs'/block;out.mkdir(parents=True,exist_ok=False)
    config=CONFIG;manifest=json.loads((Path(spec['output'])/'encoding.json').read_text())
    options=spec['training'];torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda');assert torch.cuda.get_device_name(0)==spec['required_device']
    started=time.time();training=load_data(spec,'train',manifest);validation=load_data(spec,'validation',manifest)
    n=len(training['labels']);assert n==spec['expected_training_samples']
    mean=std=None
    def standardize(data,fit=False):
        nonlocal mean,std
        mask=data['features'][...,-1:];x=transform_fmt(data['features'][...,:-1],True)
        if fit:
            valid=x[mask[...,0]>.5];mean=valid.mean(0,dtype=np.float64);std=valid.std(0,dtype=np.float64);std[std<1e-8]=1
        x=np.clip(((x-mean)/std).astype(np.float32),-8.,8.)
        data['features']=np.concatenate((x*mask,mask),-1)
    standardize(training,True);standardize(validation)
    ref=json.loads((Path(spec['output'])/'full_reference.json').read_text())
    assert np.array_equal(mean,ref['normalization']['mean']) and np.array_equal(std,ref['normalization']['std'])
    model=MaskedFMT(spec['blocks'][block],candidate['dropout']).to(device)
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
    assert abs(pipeline.binary_metrics(validation['labels'],val_probability,.5)['f1']-best[0])<1e-12
    pipeline.write(out/'selection.lock.json',dict(selected_epoch=best_epoch,criterion='validation_F1_then_AP',
        fixed_threshold=.5,test_loaded=False,identity=identity(CONFIG)))
    train_probability,_=predict(training);probabilities={}
    for split,data,p in (('train',training,train_probability),('validation',validation,val_probability)):
        for key in ('labels','flow_index','head_component','scale_id'):probabilities[split+'_'+key]=data[key]
        probabilities[split+'_probability']=p
    np.savez_compressed(out/'predictions.npz',**probabilities)
    result=dict(version=spec['version'],identity=identity(CONFIG),block=block,removed_slice=spec['blocks'][block],
        seed=seed,parameters=sum(p.numel() for p in model.parameters()),selected_epoch=best_epoch,
        stopped_epoch=history[-1]['epoch'],training=pipeline.all_metrics(training,train_probability,.5,spec),
        validation=pipeline.all_metrics(validation,val_probability,.5,spec),normalization_exactly_matches_full=True,
        predictions_sha256=pipeline.sha(out/'predictions.npz'),weights_written=0,test_loaded=False,
        device=torch.cuda.get_device_name(0),seconds=time.time()-started)
    pipeline.write(out/'result.json',result)


def merge(spec):
    root=Path(spec['output']);ref=json.loads((root/'full_reference.json').read_text());rows=[];audits=[]
    metadata={}
    for split in SPLITS:
        parts=[]
        for i,flow in enumerate(spec['flows']):
            with np.load(root/flow['name']/split/'metadata.npz') as z:
                part={k:z[k] for k in ('labels','head_component','scale_id')}
            part['flow_index']=np.full(len(part['labels']),i);parts.append(part)
        metadata[split]={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    for block in ['full',*spec['blocks']]:
        folder=root/'runs'/block; r=ref if block=='full' else json.loads((folder/'result.json').read_text())
        path=root/'full_reference_predictions.npz' if block=='full' else folder/'predictions.npz'
        if block!='full':
            assert r['predictions_sha256']==pipeline.sha(path) and r['parameters']==109634 and not r['test_loaded']
            assert r['identity']['config_sha256']==pipeline.sha(CONFIG)
            history=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()]
            best=max(history,key=lambda h:(h['validation_f1'],h['validation_average_precision']))
            assert best['epoch']==r['selected_epoch']
            rng=np.random.default_rng(spec['seed'])
            for h in history:
                order=rng.permutation(spec['expected_training_samples'])
                assert h['training_examples']==len(order) and h['permutation_sha256']==hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()
        errors=[]
        with np.load(path) as z:
            assert all(not k.startswith('test_') for k in z.files)
            for split,field in (('train','training'),('validation','validation')):
                for key,expected in metadata[split].items():assert np.array_equal(z[split+'_'+key],expected)
                y,p=z[split+'_labels'],z[split+'_probability'];calculated=metric(y,p)
                for key in calculated:errors.append(abs(calculated[key]-r[field]['pooled'][key]))
                for i,flow in enumerate(spec['flows']):
                    valid=z[split+'_flow_index']==i;fm=metric(y[valid],p[valid])
                    for key in fm:errors.append(abs(fm[key]-r[field]['per_flow'][flow['name']][key]))
        assert max(errors)<1e-12
        rows.append(dict(block=block,removed_slice=None if block=='full' else spec['blocks'][block],
            train_f1=r['training']['pooled']['f1'],validation_f1=r['validation']['pooled']['f1'],
            validation_average_precision=r['validation']['pooled']['average_precision'],
            importance_f1_drop=ref['validation']['pooled']['f1']-r['validation']['pooled']['f1'],
            per_flow_validation={f['name']:r['validation']['per_flow'][f['name']]['f1'] for f in spec['flows']},
            selected_epoch=r['selected_epoch'],reference_reused=block=='full'))
        audits.append(dict(block=block,max_metric_error=max(errors),predictions_sha256=pipeline.sha(path)))
    ranking=sorted(rows[1:],key=lambda r:r['importance_f1_drop'],reverse=True)
    assert not any(root.rglob('*.pt')) and not any(root.rglob('*.pth')) and not any(root.rglob('*.ckpt'))
    report=dict(version=spec['version'],identity=identity(CONFIG),rows=rows,ranking=ranking,
        reference_result_sha256=spec['reference_result_sha256'],seed=spec['seed'],fixed_threshold=.5,
        importance_definition='Full minus leave-one-block-out retrained validation F1',
        single_seed_exploratory=True,test_loaded=False)
    pipeline.write(root/'summary.json',report)
    pipeline.write(root/'independent_audit.json',dict(status='PASS',runs=audits,
        all_training_rows_each_epoch=True,test_loaded=False,summary_sha256=pipeline.sha(root/'summary.json')))
    print(json.dumps(ranking),flush=True)


def submit(spec,phase,dependency=None):
    from experiments.Record_Task1235_AngleFeatures_1_1 import append_record
    root=Path(spec['output']);(root/'logs').mkdir(parents=True,exist_ok=True)
    gpu=phase in ('preflight','train')
    command=['sbatch','--parsable','--cpus-per-task=4','--mem=32G','--time=01:00:00',
        '--job-name=t4c419-'+phase,'--output='+str(root/'logs'/f'{phase}.%A_%a.out'),
        '--error='+str(root/'logs'/f'{phase}.%A_%a.err')]
    if gpu:command+=['--gres=gpu:1','--constraint=v100']
    if phase=='train':command+=['--array=0-4%5']
    if dependency:command+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
    command+=['ibex_bash/task4c_fmt_v5_blocks_4p19.sh',phase]
    job=subprocess.check_output(command,text=True).strip().split(';')[0]
    row=dict(version=spec['version'],phase=phase,job_id=job,command=command,config=CONFIG,
        submitted_at_utc=datetime.now(timezone.utc).isoformat(),expected_device=spec['required_device'] if gpu else 'CPU',**identity(CONFIG))
    append_record(root/'submissions.jsonl',row,'Task4C 4.19 submitted');append_record('docs/ibex_run_registry.md',row,'Task4C 4.19 submitted')
    row.pop('source_sha256');print(json.dumps(row),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('preflight','train','merge','submit','runtime'))
    p.add_argument('--index',type=int,default=0);p.add_argument('--submit-phase');p.add_argument('--dependency')
    p.add_argument('--runtime-phase');p.add_argument('--state');p.add_argument('--exit-code',type=int)
    a=p.parse_args();spec=load_spec()
    if a.phase=='submit':submit(spec,a.submit_phase,a.dependency)
    elif a.phase=='runtime':parent.runtime(spec,CONFIG,a.runtime_phase,a.state,a.exit_code)
    elif a.phase=='train':train(spec,a.index)
    else:{'preflight':preflight,'merge':merge}[a.phase](spec)


if __name__=='__main__':main()
