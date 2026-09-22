"""Single-seed three-flow training with incremental raw-curl Couette data."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.metrics import f1_score
from experiments.Task4C_LabelSplitQuick_1_1 import sha,write,verify_sources
from experiments import Task4C_Rotation50_1_1 as old
from experiments.Task4C_CouetteTrainingData_2_1 import seed_metrics, sources_for
from FMT_Utils.Task4C_TangentJitter_1_1 import tangent_jitter,set_dropout

from experiments.Task4C_CouetteTrainingData_2_1 import prepare,encode as build_encoding

CONFIG='config/mainExp_Task4C_CouetteTraining_2.1.json'


def spec():
    s=json.loads(Path(CONFIG).read_text())
    if not s.get('protocol_confirmed') or not isinstance(s.get('fmt_augmentation'),bool):
        raise RuntimeError('Pending FMT augmentation protocol clarification')
    assert isinstance(s['dataset_audit_sha256'],str) and len(s['dataset_audit_sha256'])==64
    return s


def raw_geometry(sources,d,ids):
    p=np.empty((len(ids),7,32,3),np.float64);q=np.empty((len(ids),7,3),np.float64)
    for fi in range(len(sources)):
        local=np.flatnonzero(d['flow'][ids]==fi)
        if len(local):p[local],q[local],_=sources[fi].gather(d['sample'][ids[local]],d['length'][ids[local]])
    return torch.tensor(p,device='cuda'),torch.tensor(q,device='cuda')


def encode(s,p,q,candidate,noise=None):
    with torch.no_grad():
        if noise is not None:p=tangent_jitter(p,noise,s['augmentation']['sigma'],s['augmentation']['maximum'])
        g,ns,count=old.conv.data.old.v2.normalize_bundle(p,q)
        token,_=old.conv.data.old.fmt.encode(g,ns,count,torch.zeros(len(g),device='cuda',dtype=torch.long),candidate)
        token=token.clone();token[...,:141]=torch.sign(token[...,:141])*torch.log1p(token[...,:141].abs())
        return token


def run(index,pilot=False):
    s=spec();verify_sources();fold=index//2;name=('fmt','conv8')[index%2];out=Path(s['output']);root=Path(s['old_training_output']);prep=json.loads((out/'prepared.json').read_text());assert prep['complete']
    assert sha(Path(s['dataset_output'])/'data_audit.json')==prep['dataset_audit_sha256']
    for p,digest in prep['files'].items():
        if p.startswith(f'fold{fold}/'):assert sha(out/p)==digest
    d={r:dict(np.load(out/f'fold{fold}/dataset/{r}.npz')) for r in ('train','test')}
    baseline=prep;encoded=json.loads((out/f'fold{fold}/encoding/complete.json').read_text())
    for p,digest in {**baseline['files'],**encoded['files']}.items():
        if p.startswith(f'fold{fold}/'):assert sha(out/p)==digest
    raw={r:np.load(out/f'fold{fold}'/(f'dataset/{r}_features.npy' if name=='fmt' else f'encoding/{r}_voxels.npy'),mmap_mode='r') for r in ('train','test')}
    ss=old.source_spec(s);candidate=ss['candidates'][0];old.conv.original.old.fmt_old.baseline.deterministic('cuda')
    dest=out/('pilot' if pilot else f'fold{fold}/runs')/name;dest.mkdir(parents=True,exist_ok=False)
    aug=(name=='fmt' and s['fmt_augmentation']);points=seeds=None;sources=None
    if name=='fmt':
        v=raw['train'][:,0,:141];mean=v.mean(0,dtype=np.float64);std=v.std(0,dtype=np.float64);std[std<1e-8]=1
        norm=(torch.tensor(mean,device='cuda'),torch.tensor(std,device='cuda'),len(v));np.savez(dest/'normalizer.npz',mean=mean,std=std)
        tensors={r:old.conv.original.standardize(candidate,torch.tensor(np.array(v),device='cuda'),norm) for r,v in raw.items()}
        if aug:
            sources=sources_for(s);ids=np.concatenate([np.flatnonzero(d['train']['flow']==fi)[:6] for fi in range(3)])
            p,q=raw_geometry(sources,d['train'],ids);clean=encode(s,p,q,candidate)
            np.testing.assert_allclose(clean.cpu(),raw['train'][ids],atol=2e-4,rtol=2e-4)
            torch.testing.assert_close(clean,encode(s,p,q,candidate,torch.zeros(p.shape[:-1],device='cuda',dtype=p.dtype)),atol=0,rtol=0)
            noise=torch.ones(p.shape[:-1],device='cuda',dtype=p.dtype);assert not torch.equal(clean,encode(s,p,q,candidate,noise))
    def batch(role,ids):
        if name=='fmt':return tensors[role][ids]
        return torch.tensor(np.array(raw[role][ids]),device='cuda',dtype=torch.float32)
    def make_model():
        if name=='conv8':return old.conv.original.engine.Conv915(.15).cuda()
        model=old.conv.original.make_model(candidate)
        if s['fmt_dropout']!=.15:set_dropout(model,s['fmt_dropout'])
        return model
    torch.manual_seed(s['seed']);model=make_model();model.eval();x=batch('train',np.arange(32))
    with torch.no_grad():torch.testing.assert_close(model(x),torch.cat([model(t) for t in x.split(8)]),atol=2e-5,rtol=2e-5)
    labels=torch.tensor(d['train']['new_label'].astype(np.int64),device='cuda');model.train();pilot_x=batch('train',np.arange(128))
    if aug:
        p,q=raw_geometry(sources,d['train'],np.arange(128));noise=torch.randn(p.shape[:-1],device='cuda',dtype=p.dtype)
        pilot_x=old.conv.original.standardize(candidate,encode(s,p,q,candidate,noise),norm)
    loss=torch.nn.functional.cross_entropy(model(pilot_x),labels[:128]);loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    write(dest/'preflight.json',dict(complete=True,batch128_finite_gradient=True,batch_consistency=True,new_labels=True,augmentation=aug,gpu=torch.cuda.get_device_name()))
    if pilot:return
    if aug:
        n=len(labels);points=torch.empty((n,7,32,3),dtype=torch.float64,device='cuda');seeds=torch.empty((n,7,3),dtype=torch.float64,device='cuda')
        for first in range(0,n,2048):
            ids=np.arange(first,min(first+2048,n));p,q=raw_geometry(sources,d['train'],ids);points[first:first+len(ids)]=p;seeds[first:first+len(ids)]=q
    torch.manual_seed(s['seed']);np.random.seed(s['seed']);rng=np.random.default_rng(s['seed']);aug_rng=torch.Generator(device='cuda').manual_seed(s['seed']);model=make_model()
    initial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest();assert initial==s['initial_parameter_sha256'][name]
    optimizer=torch.optim.AdamW(model.parameters(),lr=s['learning_rate'],weight_decay=s['weight_decay'])
    weight=old.core.loss_weights(d['train']['new_label']);weights=torch.tensor(weight,device='cuda',dtype=torch.float32)
    def predict(role):
        model.eval();parts=[];bs=1024 if name=='fmt' else 256
        with torch.no_grad():
            for first in range(0,len(d[role]['sample']),bs):parts.append(model(batch(role,slice(first,first+bs))).softmax(-1)[:,1].cpu().numpy())
        return np.concatenate(parts)
    history=[];start=time.perf_counter();n=len(labels)
    for epoch in range(1,s['epochs']+1):
        order=rng.permutation(n);shuffle=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest();model.train();total=0
        for first in range(0,n,s['batch_size']):
            ids=order[first:first+s['batch_size']];gpu_ids=torch.tensor(ids,device='cuda')
            if aug:
                p=points[gpu_ids];q=seeds[gpu_ids];noise=torch.randn(p.shape[:-1],dtype=p.dtype,device='cuda',generator=aug_rng)
                x=old.conv.original.standardize(candidate,encode(s,p,q,candidate,noise),norm)
            else:x=batch('train',ids)
            optimizer.zero_grad(set_to_none=True);losses=torch.nn.functional.cross_entropy(model(x),labels[gpu_ids],reduction='none');loss=(losses*weights[labels[gpu_ids]]).mean();assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),s['clip'],error_if_nonfinite=True);optimizer.step();total+=float(loss.detach())*len(ids)
        probability=predict('test');row=dict(epoch=epoch,loss=total/n,train=seed_metrics(d['train'],predict('train')),test=seed_metrics(d['test'],probability),permutation_sha256=shuffle,seconds=time.perf_counter()-start)
        if not history or row['test']['seeds']['f1']>max(h['test']['seeds']['f1'] for h in history):torch.save(model.state_dict(),dest/'best_test_model.pt')
        history.append(row);write(dest/'progress.json',dict(complete=False,latest=row));np.savez_compressed(dest/f'predictions_epoch{epoch:02}.npz',probability=probability,**d['test']);print(json.dumps(dict(fold=fold,model=name,**row)),flush=True)
    torch.save(model.state_dict(),dest/'final_model.pt');write(dest/'result.json',dict(complete=True,fold=fold,model=name,seed=s['seed'],primary=history[-1],best_test=max(history,key=lambda h:h['test']['seeds']['f1']),history=history,
        parameters=sum(p.numel() for p in model.parameters()),initial_parameters_sha256=initial,fresh_initialization=True,loaded_old_weights=False,class_weights=weight.tolist(),augmentation=aug,dropout=s['fmt_dropout'] if name=='fmt' else .15,dataset_audit_sha256=prep['dataset_audit_sha256']))


def audit():
    from experiments.Task4C_CouetteTrainingData_2_1 import audit_results
    audit_results(spec())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','encode','pilot','train','audit'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.phase=='encode':build_encoding(a.index)
    elif a.phase in ('pilot','train'):run(a.index,a.phase=='pilot')
    else:{'prepare':prepare,'audit':audit}[a.phase]()
