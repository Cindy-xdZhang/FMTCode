"""Append raw-curl Couette rows to frozen Channel/TBL training and test rows."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from experiments.Task4C_LabelSplitQuick_1_1 import sha,write,verify_sources
from experiments import Task4C_Rotation50_1_1 as old
from experiments.Task4C_SeedRetrain_1_1 import seed_metrics as two_flow_metrics
from FMT_Utils.Task4C_V3Training_3_1 import Source as BaseSource

CONFIG='config/mainExp_Task4C_CouetteTraining_2.1.json'


class Source(BaseSource):
    def __init__(self,physical,neighbors):
        self.seeds=np.load(physical/'seeds.npy',mmap_mode='r');self.curves=np.load(physical/'curves.npy',mmap_mode='r')
        self.neighbors=np.load(neighbors/'order6.npy',mmap_mode='r');self.k=6
        with np.load(physical/'metadata.npz') as z:self.meta={k:z[k] for k in ('label','instance','fold','gt_owner','previous_label')}


def sources_for(s):
    data=Path(s['dataset_output']);cfg=json.loads((data/'dataset.json').read_text());sources=[]
    for name in ('channel','tbl'):
        sources.append(Source(Path(cfg['flows'][name]['path']),Path(s['neighbor_output'])/name))
    sources.append(Source(data/'physical/couette',data/'neighbors/couette'))
    return sources


def seed_metrics(d,p):
    result=two_flow_metrics(d,p)
    take=d['flow']==2
    if take.any():
        result['per_flow']['couette']=two_flow_metrics({k:v[take] for k,v in d.items()},p[take])['seeds']
    return result


def prepare():
    s=json.loads(Path(CONFIG).read_text());verify_sources();out=Path(s['output']);data=Path(s['dataset_output']);base=Path(s['old_training_output'])
    assert not (out/'prepared.json').exists()
    assert sha(data/'data_audit.json')==s['dataset_audit_sha256']
    audit=json.loads((data/'independent_audit.json').read_text());assert audit['complete'] and audit['source_audit_sha256']==s['dataset_audit_sha256']
    assert sha(base/'prepared.json')==s['old_prepared_sha256']
    previous=json.loads((base/'prepared.json').read_text())
    for p,h in previous['files'].items():assert sha(base/p)==h
    source=sources_for(s);new=source[2];reports={};source_hashes={}
    cfg=json.loads((data/'dataset.json').read_text())
    for fi,name in enumerate(('channel','tbl','couette')):
        physical=Path(cfg['flows'][name]['path']) if fi<2 else data/'physical/couette'
        nb=Path(s['neighbor_output'])/name if fi<2 else data/'neighbors/couette'
        if fi<2:
            frozen_audit=physical.parent.parent/'data_audit.json'
            assert sha(frozen_audit)==cfg['flows'][name]['dataset_audit_sha256']
            for file,h in json.loads(frozen_audit.read_text())['frozen_files'][name].items():assert sha(physical/file)==h
        else:
            for file,h in json.loads((data/'data_audit.json').read_text())['files'].items():assert sha(physical/file)==h
        neighbor_manifest=json.loads((nb/'manifest.json').read_text())
        assert sha(nb/'order6.npy')==neighbor_manifest['files']['order6.npy']
        for p in [physical/'seeds.npy',physical/'curves.npy',physical/'metadata.npz',nb/'order6.npy']:source_hashes[str(p)]=sha(p)
    for fold in s['active_folds']:
        dest=out/f'fold{fold}/dataset';dest.mkdir(parents=True,exist_ok=False)
        for role in ('train','test'):
            original=dict(np.load(base/f'fold{fold}/dataset/{role}.npz'))
            ids=np.load(data/f'load_indices/fold{fold}/{role}_seed_indices.npy')
            assert len(ids)>0
            appended=dict(sample=np.repeat(ids,3),flow=np.full(len(ids)*3,2,np.int8),length=np.tile(np.arange(3),len(ids)),
                          **{key:np.repeat(new.meta[meta][ids],3) for key,meta in [('instance','instance'),('fold','fold'),('new_label','label'),('gt_owner','gt_owner'),('previous_label','previous_label')]})
            assert set(original)==set(appended)
            combined={key:np.concatenate([original[key],appended[key].astype(original[key].dtype)]) for key in original}
            for fi in range(3):
                take=combined['flow']==fi;ss=combined['sample'][take]
                for key,meta in [('new_label','label'),('instance','instance'),('fold','fold')]:np.testing.assert_array_equal(combined[key][take],source[fi].meta[meta][ss])
            np.savez_compressed(dest/f'{role}.npz',**combined)
            reports[f'{fold}/{role}']=dict(old_rows=len(original['sample']),couette_rows=len(appended['sample']),total_seeds=len(combined['sample'])//3,couette_positive=int(new.meta['label'][ids].sum()),old_rows_unchanged=True)
        tr=dict(np.load(dest/'train.npz'));te=dict(np.load(dest/'test.npz'))
        for fi in range(3):assert not set(tr['instance'][tr['flow']==fi])&set(te['instance'][te['flow']==fi])
        assert len(tr['sample'])==4*len(te['sample'])
    write(out/'prepared.json',dict(complete=True,dataset_audit_sha256=s['dataset_audit_sha256'],files={str(p.relative_to(out)):sha(p) for p in out.glob('fold*/dataset/*.npz')},source_hashes=source_hashes,reports=reports,validation_samples=0,old_rows_unchanged=True))


def encode(fold):
    s=json.loads(Path(CONFIG).read_text());verify_sources();out=Path(s['output']);base=Path(s['old_training_output']);prep=json.loads((out/'prepared.json').read_text())
    sources=sources_for(s);candidate=old.source_spec(s)['candidates'][0]
    dest=out/f'fold{fold}/encoding';dest.mkdir(parents=True,exist_ok=False)
    old.conv.original.old.fmt_old.baseline.deterministic('cuda')
    def features(d,ids,device):
        parts=[]
        for fi in np.unique(d['flow'][ids]):
            take=ids[d['flow'][ids]==fi];p,q,_=sources[fi].gather(d['sample'][take],d['length'][take])
            with torch.no_grad():
                g,ns,c=old.conv.data.old.v2.normalize_bundle(torch.tensor(p,device=device),torch.tensor(q,device=device))
                t,_=old.conv.data.old.fmt.encode(g,ns,c,torch.zeros(len(g),device=device,dtype=torch.long),candidate)
                t=t.cpu().numpy();t[...,:141]=old.conv.data.old.transform_fmt(t[...,:141],True)
            parts.append(t)
        return np.concatenate(parts)
    for role in ('train','test'):
        d=dict(np.load(out/f'fold{fold}/dataset/{role}.npz'));n=len(d['sample']);oldn=prep['reports'][f'{fold}/{role}']['old_rows']
        oldenc=json.loads((base/f'fold{fold}/encoding/complete.json').read_text())
        oldvox=base/f'fold{fold}/encoding/{role}_voxels.npy'
        assert sha(oldvox)==oldenc['files'][f'fold{fold}/encoding/{role}_voxels.npy']
        feats=np.lib.format.open_memmap(out/f'fold{fold}/dataset/{role}_features.npy',mode='w+',dtype=np.float32,shape=(n,1,142))
        vox=np.lib.format.open_memmap(dest/f'{role}_voxels.npy',mode='w+',dtype=np.float16,shape=(n,4,8,8,8))
        feats[:oldn]=np.load(base/f'fold{fold}/dataset/{role}_features.npy',mmap_mode='r');vox[:oldn]=np.load(oldvox,mmap_mode='r')
        for first in range(oldn,n,128):
            ids=np.arange(first,min(first+128,n));feats[ids]=features(d,ids,'cuda');g,c=old.conv.geometry(sources,d,ids)
            with torch.no_grad():vox[ids]=old.conv.data.old.bundle_voxels(torch.tensor(g,device='cuda'),torch.tensor(c,device='cuda'),8,4).half().cpu().numpy()
        for fi in range(3):
            ids=np.flatnonzero(d['flow']==fi)[:6]
            for device in ('cpu','cuda'):np.testing.assert_allclose(features(d,ids,device),feats[ids],atol=2e-4,rtol=2e-4)
            g,c=old.conv.geometry(sources,d,ids)
            for j in range(len(ids)):np.testing.assert_allclose(vox[ids[j]].astype(np.float32),old.conv.reference_voxel(g[j],int(c[j]),8),atol=.0005,rtol=.002)
        feats.flush();vox.flush();assert np.isfinite(feats).all() and np.isfinite(vox).all()
        print(json.dumps(dict(encoded=role,fold=fold,rows=n)),flush=True)
    paths=[*dest.glob('*.npy'),* (out/f'fold{fold}/dataset').glob('*_features.npy')]
    write(dest/'complete.json',dict(complete=True,fmt_cpu_gpu_checked=True,independent_reference_voxels=True,files={str(p.relative_to(out)):sha(p) for p in paths}))


def audit_results(s):
    verify_sources();out=Path(s['output']);results={}
    prep=json.loads((out/'prepared.json').read_text())
    for p,h in prep['source_hashes'].items():assert sha(p)==h
    for fold in s['active_folds']:
        for name in ('fmt','conv8'):
            dest=out/f'fold{fold}/runs/{name}';r=json.loads((dest/'result.json').read_text());assert r['complete'] and len(r['history'])==s['epochs']
            d=dict(np.load(out/f'fold{fold}/dataset/test.npz'))
            for row in r['history']:
                with np.load(dest/f"predictions_epoch{row['epoch']:02}.npz") as p:
                    for k,v in d.items():np.testing.assert_array_equal(p[k],v)
                    assert seed_metrics(d,p['probability'])==row['test']
                    assert abs(f1_score(d['new_label'][::3],(p['probability'].reshape(-1,3)>=.5).sum(1)>=2)-row['test']['seeds']['f1'])<1e-12
            results[f'{fold}/{name}']=dict(final=r['primary'],best_test_selected=r['best_test'])
    write(out/'result_audit.json',dict(complete=True,results=results,all_epoch_test_predictions_recomputed=True,seed=s['seed'],validation_samples=0,fold4=s['unchanged_fold4']))
