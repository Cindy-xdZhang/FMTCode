"""Five original hairpin folds, two frozen architectures, one random seed."""
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
from experiments.Task4C_SeedRetrain_1_1 import seed_metrics
from FMT_Utils import Task4C_LabelSplitQuick_1_1 as core
from FMT_Utils.Task4C_Rotation50_1_1 import select_test

CONFIG = 'config/Verify_Task4C_Rotation50_1.1.json'


def spec():
    return json.loads(Path(CONFIG).read_text())


def source_spec(s):
    result = json.loads(Path(s['training_config']).read_text())
    result.update(source_output=s['source_output'], neighbor_output=s['neighbor_output'])
    return result


def prepare():
    s = spec(); verify_sources(); out = Path(s['output'])
    assert not (out / 'prepared.json').exists()
    source = Path(s['source_output']); audit = json.loads((source / 'data_audit.json').read_text())
    assert sha(source / 'data_audit.json') == s['source_audit_sha256']
    cache = Path(s['feature_cache']); raw = {}
    for role, digest in s['feature_cache_sha256'].items():
        assert sha(cache / f'{role}_features.npy') == digest
        raw[role] = np.load(cache / f'{role}_features.npy', mmap_mode='r')
    offsets = dict(train=0, test=0); reports = {}; buffers = {}
    for fold in range(5):
        buffers[fold] = {role: {key: [] for key in ('sample','flow','length','instance','fold','new_label','gt_owner','features')} for role in ('train','test')}
    for fi, name in enumerate(('channel','tbl')):
        folder = source / 'physical' / name
        for file in ('seeds.npy','curves.npy','metadata.npz'):
            assert sha(folder / file) == audit['frozen_files'][name][file]
        seeds = np.load(folder / 'seeds.npy', mmap_mode='r')
        with np.load(folder / 'metadata.npz') as z:
            meta = {k: z[k] for k in ('label','instance','fold','gt_owner','local_grid_scale','short_curve_gt_count')}
        np.testing.assert_array_equal(meta['label'], meta['short_curve_gt_count'] >= 17)
        candidate_path = Path(s['candidate_output']) / f'{name}_candidate_mask.npy'
        assert sha(candidate_path) == s['candidate_sha256'][name]
        good = np.load(candidate_path)
        features = np.empty((len(seeds),3,1,142), np.float32)
        for role in ('train','test'):
            ids = np.flatnonzero((meta['fold'] == 0) == (role == 'test')); size = len(ids) * 3
            features[ids] = raw[role][offsets[role]:offsets[role]+size].reshape(-1,3,1,142)
            offsets[role] += size
        for fold in range(5):
            test, exceptions = select_test(good,meta['fold'],meta['gt_owner'],meta['instance'],fold,s['test_points_per_flow'],s['seed'],fi,s['allow_native_head'])
            assert all(x == dict(flow=0,fold=1,instance=74) for x in exceptions)
            key = [s['seed'],fi,12] if fold == 0 else [s['seed'],fi,12,fold]
            native = np.sort(np.random.default_rng(key).choice(np.flatnonzero(meta['fold'] == fold),s['test_points_per_flow'],replace=False))
            allowed = core.distance_allowed(seeds,np.union1d(test,native),meta['local_grid_scale']) & (meta['fold'] != fold)
            key = [s['seed'],fi,13] if fold == 0 else [s['seed'],fi,13,fold]
            priority = np.random.default_rng(key).permutation(len(seeds))
            train = np.sort(priority[allowed[priority]][:s['train_points_per_flow']])
            assert len(train) == s['train_points_per_flow']
            assert not set(meta['instance'][train]) & set(meta['instance'][test])
            from scipy.spatial import cKDTree
            ratios = cKDTree(seeds[train]).query(seeds[test])[0] / meta['local_grid_scale'][test]
            assert ratios.min() >= 1 - 1e-10
            coverage = {str(int(i)): int((meta['gt_owner'][test] == i).sum()) for i in np.unique(meta['instance'][meta['fold'] == fold])}
            assert min(coverage.values()) > 0
            reports[f'{fold}/{name}'] = dict(coverage=coverage,exceptions=exceptions,train_positive=int(meta['label'][train].sum()),test_positive=int(meta['label'][test].sum()),minimum_distance_over_local_h=float(ratios.min()))
            for role, ids in (('train',train),('test',test)):
                values = dict(sample=np.repeat(ids,3),flow=np.full(len(ids)*3,fi,np.int8),length=np.tile(np.arange(3),len(ids)),
                              instance=np.repeat(meta['instance'][ids],3),fold=np.repeat(meta['fold'][ids],3),new_label=np.repeat(meta['label'][ids],3),
                              gt_owner=np.repeat(meta['gt_owner'][ids],3),features=features[ids].reshape(-1,1,142))
                for k,v in values.items(): buffers[fold][role][k].append(v)
    assert all(offsets[r] == len(raw[r]) for r in raw)
    for fold, roles in buffers.items():
        dest = out / f'fold{fold}' / 'dataset'; dest.mkdir(parents=True,exist_ok=False)
        for role, fields in roles.items():
            fields = {k:np.concatenate(v) for k,v in fields.items()}
            np.save(dest / f'{role}_features.npy', fields.pop('features'))
            np.savez_compressed(dest / f'{role}.npz', **fields)
    files = {str(p.relative_to(out)):sha(p) for p in out.glob('fold*/dataset/*')}
    write(out / 'prepared.json',dict(complete=True,files=files,per_fold_flow=reports,train_seeds=24000,test_seeds=6000,validation_seeds=0,source_audit_sha256=s['source_audit_sha256']))


def load_rows(out, fold):
    return {role:dict(np.load(out / f'fold{fold}/dataset/{role}.npz')) for role in ('train','test')}


def encode(fold):
    s = spec(); verify_sources(); out = Path(s['output']); prep = json.loads((out/'prepared.json').read_text())
    for p,digest in prep['files'].items():
        if p.startswith(f'fold{fold}/'): assert sha(out/p) == digest
    dest = out / f'fold{fold}' / 'encoding'; dest.mkdir(parents=True,exist_ok=False)
    rows = load_rows(out,fold); ss = source_spec(s); sources = [conv.data.Source(ss,fi,6) for fi in (0,1)]
    evidence = {}
    for fi,name in enumerate(('channel','tbl')):
        root = Path(s['neighbor_output']) / name; manifest = json.loads((root/'manifest.json').read_text())
        assert manifest['complete'] and manifest['source_audit_sha256'] == s['source_audit_sha256']
        assert sha(root/'order6.npy') == manifest['files']['order6.npy']
        evidence[name] = sha(root/'order6.npy')
    conv.original.old.fmt_old.baseline.deterministic('cuda')
    for role,d in rows.items():
        features = np.load(out/f'fold{fold}/dataset/{role}_features.npy',mmap_mode='r')
        for fi in (0,1):
            # Distributed probes include all three lengths and both fields.
            available = np.flatnonzero(d['flow']==fi)
            probes = np.unique(np.r_[available[:6],available[-6:],available[len(available)//2:len(available)//2+6]])
            for key,source_key in (('new_label','label'),('instance','instance'),('fold','fold')):
                np.testing.assert_array_equal(d[key][available],sources[fi].meta[source_key][d['sample'][available]])
            lines,seeds,_ = sources[fi].gather(d['sample'][probes],d['length'][probes])
            values = []
            for device in ('cpu','cuda'):
                with torch.no_grad():
                    g,ns,count=conv.data.old.v2.normalize_bundle(torch.tensor(lines,device=device),torch.tensor(seeds,device=device))
                    token,_=conv.data.old.fmt.encode(g,ns,count,torch.zeros(len(g),device=device,dtype=torch.long),ss['candidates'][0])
                    value=token.cpu().numpy();value[...,:141]=conv.data.old.transform_fmt(value[...,:141],True);values.append(value)
            for value in values: np.testing.assert_allclose(value,features[probes],atol=2e-4,rtol=2e-4)
            geometry,counts=conv.geometry(sources,d,probes)
            with torch.no_grad(): vox=conv.data.old.bundle_voxels(torch.tensor(geometry,device='cuda'),torch.tensor(counts,device='cuda'),8,4).half().float().cpu().numpy()
            for i in range(len(probes)): np.testing.assert_allclose(vox[i],conv.reference_voxel(geometry[i],int(counts[i]),8),atol=.0005,rtol=.002)
        path=dest/f'{role}_voxels.npy';n=len(d['sample'])
        arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(n,4,8,8,8))
        for first in range(0,n,128):
            ids=np.arange(first,min(first+128,n));g,c=conv.geometry(sources,d,ids)
            with torch.no_grad(): arr[ids]=conv.data.old.bundle_voxels(torch.tensor(g,device='cuda'),torch.tensor(c,device='cuda'),8,4).half().cpu().numpy()
        arr.flush(); assert np.isfinite(arr).all()
        print(json.dumps(dict(fold=fold,encoded=role,rows=n)),flush=True)
    write(dest/'complete.json',dict(complete=True,files={str(p.relative_to(out)):sha(p) for p in dest.glob('*.npy')},neighbor_hashes=evidence,fmt_cpu_gpu_checked=True,independent_reference_voxels=True))


def train(index):
    s=spec();verify_sources();fold=index//2;name=s['models'][index%2];out=Path(s['output']);prep=json.loads((out/'prepared.json').read_text())
    encoded=json.loads((out/f'fold{fold}/encoding/complete.json').read_text());assert encoded['complete']
    for p,digest in {**prep['files'],**encoded['files']}.items():
        if p.startswith(f'fold{fold}/'):assert sha(out/p)==digest
    dest=out/f'fold{fold}/runs'/name;dest.mkdir(parents=True,exist_ok=False);d=load_rows(out,fold)
    raw={r:np.load(out/f'fold{fold}'/(f'dataset/{r}_features.npy' if name=='fmt' else f'encoding/{r}_voxels.npy'),mmap_mode='r') for r in ('train','test')}
    candidate=source_spec(s)['candidates'][0]
    conv.original.old.fmt_old.baseline.deterministic('cuda')
    def make_model():return conv.original.make_model(candidate) if name=='fmt' else conv.original.engine.Conv915(s['dropout']).cuda()
    if name=='fmt':
        v=raw['train'][:,0,:141];mean=v.mean(0,dtype=np.float64);std=v.std(0,dtype=np.float64);std[std<1e-8]=1
        norm=(torch.tensor(mean,device='cuda'),torch.tensor(std,device='cuda'),len(v));np.savez(dest/'normalizer.npz',mean=mean,std=std)
        tensors={r:conv.original.standardize(candidate,torch.tensor(np.array(v),device='cuda'),norm) for r,v in raw.items()}
    def batch(role,ids):
        if name=='fmt':return tensors[role][ids]
        return torch.tensor(np.array(raw[role][ids]),device='cuda',dtype=torch.float32)
    torch.manual_seed(s['seed']);model=make_model();model.eval();x=batch('train',np.arange(32))
    with torch.no_grad():torch.testing.assert_close(model(x),torch.cat([model(v) for v in x.split(8)]),atol=2e-5,rtol=2e-5)
    labels=torch.tensor(d['train']['new_label'].astype(np.int64),device='cuda');model.train()
    loss=torch.nn.functional.cross_entropy(model(batch('train',np.arange(128))),labels[:128]);loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    write(dest/'preflight.json',dict(complete=True,batch128_finite_gradient=True,batch_consistency=True,gpu=torch.cuda.get_device_name()))
    torch.manual_seed(s['seed']);np.random.seed(s['seed']);rng=np.random.default_rng(s['seed']);model=make_model()
    initial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    assert initial==s['initial_parameter_sha256'][name]
    optimizer=torch.optim.AdamW(model.parameters(),lr=s['learning_rate'],weight_decay=s['weight_decay'])
    weight=core.loss_weights(d['train']['new_label']);weights=torch.tensor(weight,device='cuda',dtype=torch.float32)
    def predict(role):
        model.eval();parts=[];bs=1024 if name=='fmt' else 256
        with torch.no_grad():
            for first in range(0,len(d[role]['sample']),bs):parts.append(model(batch(role,slice(first,first+bs))).softmax(-1)[:,1].cpu().numpy())
        return np.concatenate(parts)
    history=[];started=time.perf_counter()
    for epoch in range(1,s['epochs']+1):
        order=rng.permutation(len(labels));shuffle=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest();model.train();total=0
        for first in range(0,len(order),s['batch_size']):
            ids=order[first:first+s['batch_size']];optimizer.zero_grad(set_to_none=True)
            losses=torch.nn.functional.cross_entropy(model(batch('train',ids)),labels[ids],reduction='none');loss=(losses*weights[labels[ids]]).mean();assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),s['clip'],error_if_nonfinite=True);optimizer.step();total+=float(loss.detach())*len(ids)
        probability=predict('test');row=dict(epoch=epoch,loss=total/len(labels),train=seed_metrics(d['train'],predict('train')),test=seed_metrics(d['test'],probability),permutation_sha256=shuffle,seconds=time.perf_counter()-started)
        if not history or row['test']['seeds']['f1']>max(h['test']['seeds']['f1'] for h in history):torch.save(model.state_dict(),dest/'best_test_model.pt')
        history.append(row);write(dest/'progress.json',dict(complete=False,latest=row));np.savez_compressed(dest/f'predictions_epoch{epoch:02}.npz',probability=probability,**d['test'])
        print(json.dumps(dict(fold=fold,model=name,**row)),flush=True)
    torch.save(model.state_dict(),dest/'final_model.pt')
    write(dest/'result.json',dict(complete=True,model=name,fold=fold,seed=s['seed'],primary=history[-1],best_test=max(history,key=lambda h:h['test']['seeds']['f1']),history=history,initial_parameters_sha256=initial,parameters=sum(p.numel() for p in model.parameters()),class_weights=weight.tolist(),fresh_initialization=True,loaded_old_weights=False,prepared_sha256=sha(out/'prepared.json')))


def audit():
    s=spec();verify_sources();out=Path(s['output']);prep=json.loads((out/'prepared.json').read_text())
    for p,digest in prep['files'].items():assert sha(out/p)==digest
    results={};pooled={name:[] for name in s['models']};all_ids=[];reference_shuffle=None
    for fold in range(5):
        d=load_rows(out,fold);tr=d['train'];te=d['test']
        assert (tr['fold']!=fold).all() and (te['fold']==fold).all()
        assert len(tr['sample'])==72000 and len(te['sample'])==18000
        all_ids.extend(zip(te['flow'][::3].tolist(),te['sample'][::3].tolist()))
        for fi,name in enumerate(('channel','tbl')):
            assert not set(tr['instance'][tr['flow']==fi]) & set(te['instance'][te['flow']==fi])
            for inst,count in prep['per_fold_flow'][f'{fold}/{name}']['coverage'].items():assert int(((te['gt_owner'][::3]==int(inst))&(te['flow'][::3]==fi)).sum())==count and count>0
        for name in s['models']:
            dest=out/f'fold{fold}/runs'/name;r=json.loads((dest/'result.json').read_text());assert r['complete'] and r['fresh_initialization'] and not r['loaded_old_weights']
            assert len(r['history'])==s['epochs']
            shuffle=[h['permutation_sha256'] for h in r['history']]
            if reference_shuffle is None:reference_shuffle=shuffle
            assert shuffle==reference_shuffle
            for row in r['history']:
                with np.load(dest/f"predictions_epoch{row['epoch']:02}.npz") as p:
                    for k,v in te.items():np.testing.assert_array_equal(p[k],v)
                    assert seed_metrics(te,p['probability'])==row['test']
                    assert abs(f1_score(te['new_label'][::3],(p['probability'].reshape(-1,3)>=.5).sum(1)>=2)-row['test']['seeds']['f1'])<1e-12
            assert r['best_test']==max(r['history'],key=lambda h:h['test']['seeds']['f1'])
            results[f'{fold}/{name}']=dict(final=r['primary'],best_test=r['best_test']);pooled[name].append(r['primary']['test']['seeds'])
    assert len(all_ids)==len(set(all_ids))==30000
    aggregate={}
    for name,items in pooled.items():
        counts={k:sum(x[k] for x in items) for k in ('tp','fp','fn','tn','samples','positive')};den=2*counts['tp']+counts['fp']+counts['fn']
        aggregate[name]=dict(**counts,pooled_f1=2*counts['tp']/max(den,1),mean_fold_f1=float(np.mean([x['f1'] for x in items])),std_across_folds=float(np.std([x['f1'] for x in items])))
    write(out/'result_audit.json',dict(complete=True,results=results,aggregate=aggregate,all_500_predictions_checked=True,test_seed_ids_disjoint_between_folds=True,all_head_instances_covered=True,same_shuffle=True,validation_points=0,seed=s['seed'],selection='epoch50 final aggregate; best-test epochs are separately test-selected, not independent final scores'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','encode','train','audit'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.phase=='encode':encode(a.index)
    elif a.phase=='train':train(a.index)
    else:{'prepare':prepare,'audit':audit}[a.phase]()
