"""Validation-only search followed by separately authorized frozen test evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256, write_json
from FMT_Utils.PrimitiveVAE_3D import geometry_metrics
from FMT_Utils.Task6Scarce_3D import subset_order, inputs_numpy, fit_small
from experiments.Task6_PrimitiveVAE_2_1 import provenance, predict

DEFAULT = 'config/Verify_Task6_ScarceGeneralization_4.1.json'


def raw_candidates(spec):
    return [dict(c, id='r'+str(i), frequencies=16) for i,c in enumerate(spec['candidates'][:4])]


def prepare(spec, config, index):
    dataset = spec['datasets'][index]
    source = Path(spec['source_cache'])
    assert sha256(source/'config.frozen.json') == spec['source_config_sha256']
    folder = Path(spec['output_root'])/'data'/dataset
    folder.mkdir(parents=True, exist_ok=False)
    src = source/'data'/dataset
    manifest = json.loads((src/'manifest.json').read_text())
    audit = json.loads((src/'audit.json').read_text())
    assert audit['passed']
    train_path, validation_path = src/'train.npz', src/'validation.npz'
    assert sha256(train_path) == manifest['files']['train']['sha256']
    assert sha256(validation_path) == manifest['files']['validation']['sha256']
    write_json(folder/'started.json', dict(provenance=provenance(config),dataset=dataset))
    files = {}
    with np.load(train_path) as cache:
        geometry = cache['geometry']
        scale = cache['scale_id']
        identities = cache['primitive_id']
        for seed in [spec['search_seed']] + spec['seeds']:
            selected = subset_order(len(geometry),seed)[:max(spec['train_sizes'])]
            target = folder/f'train_{seed}.npz'
            np.savez(target, geometry=geometry[selected], scale_id=scale[selected],
                primitive_id=identities[selected], source_indices=selected)
            files[target.name] = sha256(target)
    with np.load(validation_path) as cache:
        np.savez(folder/'validation.npz',geometry=cache['geometry'],scale_id=cache['scale_id'],
            primitive_id=cache['primitive_id'])
    files['validation.npz'] = sha256(folder/'validation.npz')
    write_json(folder/'manifest.json',dict(dataset=dataset,provenance=provenance(config),
        source_manifest_sha256=sha256(src/'manifest.json'),source_train_sha256=manifest['files']['train']['sha256'],
        source_validation_sha256=manifest['files']['validation']['sha256'],files=files,
        policy='Random nested prefixes from train only; no test file opened; both arms share identities',
        train_sizes=spec['train_sizes'],seeds=[spec['search_seed']]+spec['seeds']))


def setup(spec, config, index, phase):
    dataset,n = [(d,n) for d in spec['datasets'] for n in spec['train_sizes']][index]
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('Research training requires a GPU')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    root=Path(spec['output_root'])
    folder=root/phase/dataset/str(n)
    folder.mkdir(parents=True,exist_ok=True)
    metadata=dict(dataset=dataset,train_size=n,provenance=provenance(config),device=torch.cuda.get_device_name(0),
        phase=phase,experiment=spec['experiment'])
    write_json(folder/'started.json',metadata)
    data=root/'data'/dataset
    manifest=json.loads((data/'manifest.json').read_text())
    assert manifest['provenance']['config_sha256']==sha256(config)
    with np.load(data/'validation.npz') as cache:
        validation=cache['geometry'].copy()
    return dataset,n,folder,metadata,data,manifest,validation,torch.device('cuda')


def load_subset(data,manifest,seed,n):
    source=data/f'train_{seed}.npz'
    assert sha256(source)==manifest['files'][source.name]
    with np.load(source) as cache:
        y=cache['geometry'][:n].copy()
        ids=cache['source_indices'][:n].copy()
    assert len(y)==n and len(set(ids.tolist()))==n
    return y,hashlib.sha256(ids.tobytes()).hexdigest()


def search(spec,config,index):
    dataset,n,folder,metadata,data,manifest,vy,device=setup(spec,config,index,'search')
    seed=spec['search_seed']
    y,subset_hash=load_subset(data,manifest,seed,n)
    rows=[]
    for arm,candidates in [('raw_vae',raw_candidates(spec)),('signed_fmt_vae',spec['candidates'])]:
        for candidate in candidates:
            fit_dir=folder/(arm+'_'+candidate['id'])
            model,result=fit_small(y,vy,arm,candidate,spec['training'],seed,device,fit_dir)
            rows.append(dict(arm=arm,candidate=candidate['id'],validation_rmse_r=result['validation_rmse_r'],
                initialization_rmse_r=result['curve'][0]['validation_rmse_r'],selected_step=result['selected_step'],
                train_rmse_r=result['train_rmse_r']))
            del model
            torch.cuda.empty_cache()
    write_json(folder/'result.json',dict(**metadata,seed=seed,subset_sha256=subset_hash,rows=rows,
        test_read=False,selection_input='training and validation only'))


def select(spec,config):
    root=Path(spec['output_root'])
    primary=[]
    summaries=[]
    for dataset in spec['datasets']:
        for n in spec['train_sizes']:
            result=json.loads((root/'search'/dataset/str(n)/'result.json').read_text())
            assert result['test_read'] is False
            assert result['provenance']['config_sha256']==sha256(config)
            for row in result['rows']:
                item=dict(dataset=dataset,train_size=n,**row)
                summaries.append(item)
                if n==spec['primary_train_size']:
                    primary.append(item)
    scores={}
    for arm,candidates in [('raw_vae',raw_candidates(spec)),('signed_fmt_vae',spec['candidates'])]:
        for c in candidates:
            values=[r['validation_rmse_r'] for r in primary if r['arm']==arm and r['candidate']==c['id']]
            assert len(values)==len(spec['datasets'])
            scores[arm+'_'+c['id']]=float(np.mean(values))
    selected=min(spec['candidates'],key=lambda c:(scores['signed_fmt_vae_'+c['id']],c['id']))
    selected_raw=min(raw_candidates(spec),key=lambda c:(scores['raw_vae_'+c['id']],c['id']))
    raw_ref=raw_candidates(spec)[0]
    raw_matched=raw_candidates(spec)[selected['regularizer_index']]
    choice=dict(provenance=provenance(config),experiment=spec['experiment'],primary_train_size=spec['primary_train_size'],
        selection='minimum equal-flow mean validation RMSE at preregistered primary size; common candidate for all flows and sizes',
        fmt=selected,raw_frozen=raw_ref,raw_matched=raw_matched,raw_selected=selected_raw,scores=scores,
        rows=summaries,test_read=False)
    write_json(root/'selection.json',choice)
    print(json.dumps({k:v for k,v in choice.items() if k!='rows'},indent=2),flush=True)


def final(spec,config,index):
    root=Path(spec['output_root'])
    assert sha256(root/'selection.json')==sha256(root/'selection.before_test.json')
    selection=json.loads((root/'selection.json').read_text())
    assert selection['test_read'] is False and selection['provenance']['config_sha256']==sha256(config)
    dataset,n,folder,metadata,data,manifest,vy,device=setup(spec,config,index,'final')
    plan=[('fmt', 'signed_fmt_vae', selection['fmt'])]
    plan += [(role,'raw_vae',selection[role]) for role in ('raw_frozen','raw_matched','raw_selected')]
    unique={}
    for role,arm,candidate in plan:
        unique.setdefault((arm,candidate['id']),dict(arm=arm,candidate=candidate,roles=[]))['roles'].append(role)
    source=Path(spec['source_cache'])/'data'/dataset
    source_manifest=json.loads((source/'manifest.json').read_text())
    all_rows=[]
    fits=[]
    for seed in spec['seeds']:
        y,subset_hash=load_subset(data,manifest,seed,n)
        for item in unique.values():
            fit_dir=folder/str(seed)/(item['arm']+'_'+item['candidate']['id'])
            model,training=fit_small(y,vy,item['arm'],item['candidate'],spec['training'],seed,device,fit_dir)
            if training['validation_rmse_r']>=spec['validation_gate_rmse_r']:
                write_json(fit_dir/'validation_failed.json',training)
                raise RuntimeError('Validation remains an engineering failure; test not read for this model')
            for role in ('test','unseen_scale'):
                target=source/f'{role}.npz'
                assert sha256(target)==source_manifest['files'][role]['sha256']
                with np.load(target) as cache:
                    truth,scales=cache['geometry'].copy(),cache['scale_id'].copy()
                inputs=inputs_numpy(truth,item['arm'],item['candidate']['frequencies'])
                prediction=predict(model,inputs,device)
                np.savez(fit_dir/f'{role}_predictions.npz',prediction=prediction)
                for sid in [-1]+sorted(np.unique(scales).tolist()):
                    mask=np.ones(len(truth),bool) if sid==-1 else scales==sid
                    metric=geometry_metrics(prediction[mask],truth[mask])
                    all_rows.append(dict(seed=seed,arm=item['arm'],candidate=item['candidate']['id'],
                        comparisons=item['roles'],role=role,scale_id=sid,**metric))
            fits.append(dict(seed=seed,arm=item['arm'],candidate=item['candidate'],roles=item['roles'],
                subset_sha256=subset_hash,selected_step=training['selected_step'],
                validation_rmse_r=training['validation_rmse_r'],train_rmse_r=training['train_rmse_r']))
            del model
            torch.cuda.empty_cache()
    write_json(folder/'result.json',dict(**metadata,selection_sha256=sha256(root/'selection.json'),
        fits=fits,metrics=all_rows,model_files='none; model selection states only in RAM'))


def audit_results(spec,config):
    root=Path(spec['output_root'])
    assert sha256(root/'selection.json')==sha256(root/'selection.before_test.json')
    rows=[]
    model_count=0
    for dataset in spec['datasets']:
        source=Path(spec['source_cache'])/'data'/dataset
        for n in spec['train_sizes']:
            folder=root/'final'/dataset/str(n)
            result=json.loads((folder/'result.json').read_text())
            assert result['provenance']['config_sha256']==sha256(config)
            assert result['selection_sha256']==sha256(root/'selection.json')
            model_count+=len(result['fits'])
            for fit in result['fits']:
                seed,arm,candidate=fit['seed'],fit['arm'],fit['candidate']['id']
                model_dir=folder/str(seed)/(arm+'_'+candidate)
                for role in ('test','unseen_scale'):
                    with np.load(source/f'{role}.npz') as cache:
                        y,scales=cache['geometry'],cache['scale_id']
                    with np.load(model_dir/f'{role}_predictions.npz') as cache:
                        prediction=cache['prediction']
                    records=[m for m in result['metrics'] if (m['seed'],m['arm'],m['candidate'],m['role'])==(seed,arm,candidate,role)]
                    assert len(records)==1+len(np.unique(scales))
                    for metric in records:
                        sid=metric['scale_id']
                        mask=np.ones(len(y),bool) if sid==-1 else scales==sid
                        fresh=geometry_metrics(prediction[mask],y[mask])
                        diff=prediction[mask].astype(np.float64)[:,:,1:]-y[mask].astype(np.float64)[:,:,1:]
                        independent=float(np.sqrt(np.sum(diff*diff)/(len(diff)*7*31)))
                        np.testing.assert_allclose(independent,fresh['position_rmse_r'],rtol=1e-12)
                        for k,v in fresh.items():
                            np.testing.assert_allclose(metric[k],v,rtol=1e-10,atol=1e-12)
                        for comparison in fit['roles']:
                            rows.append(dict(dataset=dataset,train_size=n,seed=seed,comparison=comparison,
                                role=role,scale_id=sid,**{k:v for k,v in fresh.items() if not isinstance(v,list)}))
    for suffix in ('*.pt','*.pth','*.ckpt'):
        assert not any(root.rglob(suffix))
    with (root/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(root/'final_audit.json',dict(passed=True,provenance=provenance(config),models=model_count,
        rows=len(rows),selection_sha256=sha256(root/'selection.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['prepare','search','select','final','audit'])
    p.add_argument('--config',default=DEFAULT)
    p.add_argument('--index',type=int,default=0)
    args=p.parse_args();spec=json.loads(Path(args.config).read_text())
    if args.phase in ('select','audit'):
        {'select':select,'audit':audit_results}[args.phase](spec,args.config)
    else:
        {'prepare':prepare,'search':search,'final':final}[args.phase](spec,args.config,args.index)
