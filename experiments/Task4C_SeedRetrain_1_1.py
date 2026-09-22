"""Fresh single-seed fits with fixed test indices and seed-majority evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.metrics import f1_score
from experiments.Task4C_LabelSplitQuick_1_1 import sha, write, verify_sources
from experiments import Task4C_ConvQuick_1_1 as conv
from FMT_Utils import Task4C_LabelSplitQuick_1_1 as core

CONFIG='config/Verify_Task4C_SeedRetrain_1.1.json'


def spec():return json.loads(Path(CONFIG).read_text())


def rows():
    s=spec();root=Path(s['baseline_output']);record=json.loads((root/'dataset.json').read_text())
    assert sha(root/'dataset.json')==s['baseline_dataset_sha256']
    result={}
    for role,source in s['roles'].items():
        result[role]={}
        for key in ('sample','flow','length','instance','fold','new_label'):
            rel=f'dataset/{source}/{key}.npy';assert sha(root/rel)==record['files'][rel];result[role][key]=np.load(root/rel)
    return result,record


def seed_metrics(d,p):
    np.testing.assert_array_equal(d['length'],np.tile(np.arange(3),len(p)//3))
    for key in ('sample','flow','instance','new_label'):
        a=d[key].reshape(-1,3);np.testing.assert_array_equal(a,np.repeat(a[:,:1],3,1))
    votes=(p.reshape(-1,3)>=.5).sum(1)>=2;y=d['new_label'][::3];flow=d['flow'][::3]
    def metrics(y,p):
        y=np.asarray(y,bool);p=np.asarray(p,bool);tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
        return dict(samples=len(y),positive=int(y.sum()),tp=tp,fp=fp,fn=fn,tn=tn,f1=2*tp/max(2*tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))
    return dict(seeds=metrics(y,votes),per_flow={name:metrics(y[flow==fi],votes[flow==fi]) for fi,name in enumerate(('channel','tbl'))},
        rows=core.score(d['new_label'],p))


def prepare():
    s=spec();verify_sources();out=Path(s['output']);assert not (out/'prepared.json').exists()
    assert sha(s['mask'])==s['mask_sha256'] and sha(s['mask_result'])==s['mask_result_sha256']
    assert sha(Path(s['baseline_output'])/'result_audit.json')==s['baseline_audit_sha256']
    assert sha(Path(s['conv_output'])/'result_audit.json')==s['conv_audit_sha256']
    d,record=rows();mask=dict(np.load(s['mask']));keep=mask['keep_row'];assert int(keep.sum())==15378
    np.testing.assert_array_equal(d['test']['sample'][::3],mask['sample']);np.testing.assert_array_equal(d['test']['flow'][::3],mask['flow'])
    np.testing.assert_array_equal(keep,np.repeat(mask['keep_seed'],3))
    coverage=json.loads(Path(s['mask_result']).read_text())['coverage']
    for c in coverage:
        fi=('channel','tbl').index(c['flow']);actual=int(((mask['flow']==fi)&(mask['gt_owner']==c['instance'])&mask['keep_seed']).sum())
        assert actual==c['after'] and actual>0
    inputs={}
    for role,source in s['roles'].items():
        rel=f'dataset/{source}/features.npy';assert sha(Path(s['baseline_output'])/rel)==record['files'][rel];inputs[rel]=record['files'][rel]
    for model in ('conv8','conv16'):
        for role in ('train','test'):
            path=Path(s['conv_output'])/'runs'/model/f'{role}_voxels.npy';assert sha(path)==s['cache_sha256'][model][role];inputs[str(path)]=s['cache_sha256'][model][role]
    dest=out/'dataset';dest.mkdir(parents=True,exist_ok=False)
    for role,data in d.items():
        take=np.arange(len(data['sample'])) if role=='train' else np.flatnonzero(keep)
        np.savez_compressed(dest/f'{role}.npz',source_rows=take,**{k:v[take] for k,v in data.items()})
    tr=dict(np.load(dest/'train.npz'));te=dict(np.load(dest/'test.npz'))
    assert len(tr['sample'])==72000 and (tr['fold']!=0).all() and (te['fold']==0).all()
    for fi in (0,1):assert not set(tr['instance'][tr['flow']==fi])&set(te['instance'][te['flow']==fi])
    write(out/'prepared.json',dict(complete=True,input_files=inputs,coverage=coverage,mask_sha256=s['mask_sha256'],
        train_seeds=len(tr['sample'])//3,test_seeds=len(te['sample'])//3,files={str(p.relative_to(out)):sha(p) for p in dest.iterdir()}))


def train(index):
    s=spec();verify_sources();out=Path(s['output']);prep=json.loads((out/'prepared.json').read_text());assert prep['complete']
    for p,digest in prep['files'].items():assert sha(out/p)==digest
    name=s['models'][index];dest=out/'runs'/name;dest.mkdir(parents=True,exist_ok=False)
    d={role:dict(np.load(out/'dataset'/f'{role}.npz')) for role in ('train','test')}
    raw={};candidate=json.loads(Path(s['training_config']).read_text())['candidates'][0]
    for role,source in s['roles'].items():
        if name=='fmt':path=Path(s['baseline_output'])/'dataset'/source/'features.npy';digest=prep['input_files'][f'dataset/{source}/features.npy']
        else:path=Path(s['conv_output'])/'runs'/name/f'{role}_voxels.npy';digest=s['cache_sha256'][name][role]
        assert sha(path)==digest;raw[role]=np.load(path,mmap_mode='r')
    conv.original.old.fmt_old.baseline.deterministic('cuda')
    def make_model():return conv.original.make_model(candidate) if name=='fmt' else conv.original.engine.Conv915(.15).cuda()
    if name=='fmt':
        v=raw['train'][:,0,:141];mean=v.mean(0,dtype=np.float64);std=v.std(0,dtype=np.float64);std[std<1e-8]=1
        norm=(torch.tensor(mean,device='cuda'),torch.tensor(std,device='cuda'),len(v));np.savez(dest/'normalizer.npz',mean=mean,std=std)
        tensors={r:conv.original.standardize(candidate,torch.tensor(np.array(v[d[r]['source_rows']]),device='cuda'),norm) for r,v in raw.items()}
    def batch(role,ids):
        if name=='fmt':return tensors[role][ids]
        return torch.tensor(np.array(raw[role][d[role]['source_rows'][ids]]),device='cuda',dtype=torch.float32)
    # Fresh numerical preflight; no old checkpoint is loaded.
    torch.manual_seed(s['seed']);model=make_model();model.eval();x=batch('train',np.arange(32))
    with torch.no_grad():torch.testing.assert_close(model(x),torch.cat([model(v) for v in x.split(8)]),atol=2e-5,rtol=2e-5)
    labels=torch.tensor(d['train']['new_label'].astype(np.int64),device='cuda');model.train()
    loss=torch.nn.functional.cross_entropy(model(batch('train',np.arange(128))),labels[:128]);loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    write(dest/'preflight.json',dict(complete=True,batch128_finite_gradient=True,batch_consistency=True,gpu=torch.cuda.get_device_name(),cache_hashes_verified=True))
    torch.manual_seed(s['seed']);np.random.seed(s['seed']);rng=np.random.default_rng(s['seed']);model=make_model()
    initial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    reference_root=Path(s['baseline_output'])/'runs/v2' if name=='fmt' else Path(s['conv_output'])/'runs'/name
    reference=json.loads((reference_root/'result.json').read_text());assert initial==reference['initial_parameters_sha256']
    optimizer=torch.optim.AdamW(model.parameters(),lr=s['learning_rate'],weight_decay=s['weight_decay'])
    weight=core.loss_weights(d['train']['new_label']);np.testing.assert_array_equal(weight,reference['class_weights'])
    weights=torch.tensor(weight,device='cuda',dtype=torch.float32)
    def predict(role):
        model.eval();parts=[];bs=1024 if name=='fmt' else 256
        with torch.no_grad():
            for first in range(0,len(d[role]['sample']),bs):parts.append(model(batch(role,slice(first,first+bs))).softmax(-1)[:,1].cpu().numpy())
        return np.concatenate(parts)
    history=[];started=time.perf_counter()
    for epoch in range(1,s['epochs']+1):
        order=rng.permutation(len(labels));shuffle=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest();assert shuffle==reference['history'][epoch-1]['permutation_sha256']
        model.train();total=0
        for first in range(0,len(order),s['batch_size']):
            ids=order[first:first+s['batch_size']];optimizer.zero_grad(set_to_none=True)
            losses=torch.nn.functional.cross_entropy(model(batch('train',ids)),labels[ids],reduction='none');loss=(losses*weights[labels[ids]]).mean();assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),s['clip'],error_if_nonfinite=True);optimizer.step();total+=float(loss.detach())*len(ids)
        probability=predict('test');train_probability=predict('train')
        row=dict(epoch=epoch,loss=total/len(labels),train=seed_metrics(d['train'],train_probability),test=seed_metrics(d['test'],probability),
            permutation_sha256=shuffle,seconds=time.perf_counter()-started)
        history.append(row);write(dest/'progress.json',dict(complete=False,latest=row))
        np.savez_compressed(dest/f'predictions_epoch{epoch:02}.npz',probability=probability,**d['test'])
        print(json.dumps(dict(model=name,**row)),flush=True)
    torch.save(model.state_dict(),dest/'final_model.pt')
    write(dest/'result.json',dict(complete=True,model=name,seed=s['seed'],primary=history[-1],history=history,initial_parameters_sha256=initial,
        parameters=sum(p.numel() for p in model.parameters()),class_weights=weight.tolist(),fresh_initialization=True,loaded_old_weights=False,
        prepared_sha256=sha(out/'prepared.json'),evaluation_provenance=s['evaluation_provenance']))


def audit():
    s=spec();verify_sources();out=Path(s['output']);d=dict(np.load(out/'dataset/test.npz'));results={}
    for name in s['models']:
        dest=out/'runs'/name;r=json.loads((dest/'result.json').read_text());assert r['complete'] and r['fresh_initialization'] and not r['loaded_old_weights']
        for row in r['history']:
            with np.load(dest/f"predictions_epoch{row['epoch']:02}.npz") as p:
                for k,v in d.items():np.testing.assert_array_equal(p[k],v)
                assert seed_metrics(d,p['probability'])==row['test']
                y=d['new_label'][::3];vote=(p['probability'].reshape(-1,3)>=.5).sum(1)>=2
                assert abs(f1_score(y,vote)-row['test']['seeds']['f1'])<1e-12
        results[name]=r['primary']
    write(out/'result_audit.json',dict(complete=True,results=results,all_30_predictions_checked=True,all_fresh_initialization=True,
        primary_metric=s['primary_metric'],evaluation_provenance=s['evaluation_provenance'],mask_sha256=s['mask_sha256']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','train','audit'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    train(a.index) if a.phase=='train' else {'prepare':prepare,'audit':audit}[a.phase]()
