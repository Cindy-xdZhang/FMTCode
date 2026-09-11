"""Versioned FMT/geometry expert comparisons and self-supervised reconstruction."""
import argparse
import copy
import csv
import datetime
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256,write_json
from FMT_Utils.Task6FMTGeometryMoE_3D import train_moe,predict,original_fmt
from FMT_Utils.PrimitiveVAE_3D import fmt_tokens
from FMT_Utils.Task6PNNTrans_3D import train_model as train_frozen, predict as predict_frozen
from experiments.Task6_DirectNeural_5_1 import read,provenance,prepare as prepare_flow,load_data,gpu
from experiments.Task6_PNNTrans_1_1 import evaluate,load_truth

DEFAULT = 'config/Verify_Task6_FMTGeometryMoE_1.1.json'


def prepare(spec,config,index=0):
    reference = Path(spec['baseline_cache'])
    assert sha256(reference/'config.frozen.json')==spec['baseline_config_sha256']
    assert sha256(reference/'selection.json')==spec['baseline_selection_sha256']
    assert sha256(reference/'selection.before_test.json')==spec['baseline_selection_sha256']
    rows=[]
    for i,dataset in enumerate(spec['datasets']):
        prepare_flow(spec,config,i)
        y,_,ids = load_data(spec,config,dataset,spec['search_seed'],spec['primary_train_size'],validation=False)
        baseline = read(reference/'baseline'/dataset/'result.json')
        assert baseline['train_size']==len(y) and baseline['subset_sha256']==ids and not baseline['test_read']
        assert baseline['provenance']['config_sha256']==spec['baseline_config_sha256']
        # Parity is checked on the first training entries, never chosen by a label/error.
        np.testing.assert_allclose(original_fmt(torch.tensor(y[:8])).numpy(),fmt_tokens(y[:8]),rtol=0,atol=0)
        rows.append(baseline)
    write_json(Path(spec['output_root'])/'baseline_reference.json',dict(provenance=provenance(config),
        rows=rows,source_selection_sha256=spec['baseline_selection_sha256'],test_read=False,train_size=spec['primary_train_size']))


def fit_check(spec,config,index):
    dataset=spec['datasets'][index]
    y,_,ids=load_data(spec,config,dataset,spec['search_seed'],spec['fit_check']['samples'],validation=False)
    device=gpu()
    rows=[]
    for family in spec['families']:
        c=copy.deepcopy(next(c for c in spec['candidates'] if c['geometry_branch']==family))
        c.update(learning_rate=.001,dropout=0.,weight_decay=0.,auxiliary_weight=0.)
        model,result=train_moe(y,y,'fmt_moe',c,spec['fit_check'],spec['search_seed'],device,
            Path(spec['output_root'])/'fit_check'/dataset/family)
        ratio=result['train_rmse_r']/max(result['curve'][0]['train_rmse_r'],1e-12)
        zero_ratio=result['train_zero_latent_rmse_r']/max(result['train_rmse_r'],1e-12)
        rows.append(dict(family=family,error_ratio=ratio,zero_latent_ratio=zero_ratio,
            train_rmse_r=result['train_rmse_r'],passed=ratio<spec['fit_check']['maximum_error_ratio']
            and zero_ratio>spec['fit_check']['minimum_zero_latent_error_ratio']))
        del model
        torch.cuda.empty_cache()
    passed=all(r['passed'] for r in rows)
    write_json(Path(spec['output_root'])/'fit_check'/dataset/'result.json',dict(provenance=provenance(config),
        dataset=dataset,rows=rows,passed=passed,subset_sha256=ids,test_read=False))
    assert passed,'Training-only small-set memorization failed'


def metric_row(result):
    return {k:result[k] for k in ('arm','train_rmse_r','validation_rmse_r','selected_step',
                                'validation_gates','validation_branch_diagnostics','structure')}


def screen(spec,config,index):
    dataset,candidate=[(d,c) for d in spec['screen_datasets'] for c in spec['candidates']][index]
    assert read(Path(spec['output_root'])/'fit_check'/dataset/'result.json')['passed']
    y,vy,ids=load_data(spec,config,dataset,spec['search_seed'],spec['primary_train_size'])
    folder=Path(spec['output_root'])/'screen'/dataset/candidate['id']
    model,result=train_moe(y,vy,'fmt_moe',candidate,spec['screening'],spec['search_seed'],gpu(),folder/'fmt_moe')
    write_json(folder/'result.json',dict(provenance=provenance(config),dataset=dataset,candidate=candidate['id'],
        family=candidate['geometry_branch'],subset_sha256=ids,test_read=False,**metric_row(result)))
    del model


def ranking(spec,rows,baseline):
    reference={r['dataset']:r['validation_rmse_r'] for r in baseline}
    scores={}
    for c in spec['candidates']:
        selected=[r for r in rows if r['candidate']==c['id']]
        assert sorted(r['dataset'] for r in selected)==sorted(spec['screen_datasets'])
        scores[c['id']]=float(np.mean([r['validation_rmse_r']/reference[r['dataset']] for r in selected]))
    candidates=[]
    for family in spec['families']:
        candidates.extend(sorted([c for c in spec['candidates'] if c['geometry_branch']==family],
                                 key=lambda c:(scores[c['id']],c['id']))[:spec['promote_per_family']])
    return scores,candidates


def promote(spec,config,index=0):
    root=Path(spec['output_root'])
    assert not (root/'promotion.json').exists()
    rows=[]
    for dataset in spec['screen_datasets']:
        for c in spec['candidates']:
            row=read(root/'screen'/dataset/c['id']/'result.json')
            assert row['provenance']['config_sha256']==sha256(config) and not row['test_read']
            rows.append(row)
    scores,candidates=ranking(spec,rows,read(root/'baseline_reference.json')['rows'])
    write_json(root/'promotion.json',dict(provenance=provenance(config),rows=rows,scores=scores,candidates=candidates,test_read=False))
    shutil.copyfile(root/'promotion.json',root/'promotion.frozen.json')


def promoted(spec,config):
    root=Path(spec['output_root'])
    assert sha256(root/'promotion.json')==sha256(root/'promotion.frozen.json')
    data=read(root/'promotion.json')
    assert data['provenance']['config_sha256']==sha256(config) and not data['test_read']
    return data['candidates']


def refine(spec,config,index):
    dataset,candidate=[(d,c) for d in spec['datasets'] for c in promoted(spec,config)][index]
    root=Path(spec['output_root'])
    y,vy,ids=load_data(spec,config,dataset,spec['search_seed'],spec['primary_train_size'])
    rows=[]
    device=gpu()
    folder=root/'refine'/dataset/candidate['id']
    for arm in spec['arms']:
        model,result=train_moe(y,vy,arm,candidate,spec['training'],spec['search_seed'],device,folder/arm)
        rows.append(metric_row(result))
        del model
        torch.cuda.empty_cache()
    assert rows[0]['structure']['trainable_parameters']==rows[1]['structure']['trainable_parameters']
    write_json(folder/'result.json',dict(provenance=provenance(config),dataset=dataset,candidate=candidate['id'],
        family=candidate['geometry_branch'],rows=rows,subset_sha256=ids,test_read=False,promotion_sha256=sha256(root/'promotion.json')))


def select(spec,config,index=0):
    root=Path(spec['output_root'])
    assert not (root/'selection.json').exists()
    candidates=promoted(spec,config)
    rows=[]
    for dataset in spec['datasets']:
        for c in candidates:
            data=read(root/'refine'/dataset/c['id']/'result.json')
            assert data['provenance']['config_sha256']==sha256(config) and not data['test_read']
            assert data['promotion_sha256']==sha256(root/'promotion.json')
            assert sorted(r['arm'] for r in data['rows'])==sorted(spec['arms'])
            rows.extend(dict(dataset=dataset,candidate=c['id'],family=c['geometry_branch'],**r) for r in data['rows'])
    baseline=read(root/'baseline_reference.json')['rows']
    chosen={}
    for family in spec['families']:
        group=[c for c in candidates if c['geometry_branch']==family]
        scores={c['id']:float(np.mean([r['validation_rmse_r'] for r in rows if r['candidate']==c['id'] and r['arm']=='fmt_moe'])) for c in group}
        best=min(group,key=lambda c:(scores[c['id']],c['id']))
        selected=[r for r in rows if r['candidate']==best['id']]
        assert len(selected)==len(spec['datasets'])*len(spec['arms'])
        means={a:float(np.mean([r['validation_rmse_r'] for r in selected if r['arm']==a])) for a in spec['arms']}
        means['raw_frozen']=float(np.mean([r['validation_rmse_r'] for r in baseline]))
        failed=[r for r in selected if not np.isfinite(r['validation_rmse_r']) or r['validation_rmse_r']>=spec['validation_gate_rmse_r']]
        chosen[family]=dict(candidate=best,scores=scores,means=means,failures=failed,qualified=not failed)
    write_json(root/'selection.json',dict(provenance=provenance(config),families=chosen,rows=rows,
        baseline_rows=baseline,test_read=False,gate_passed=any(x['qualified'] for x in chosen.values()),rule=spec['selection']))
    shutil.copyfile(root/'selection.json',root/'selection.before_test.json')


def frozen_selection(spec,config):
    root=Path(spec['output_root'])
    assert sha256(root/'selection.json')==sha256(root/'selection.before_test.json')
    data=read(root/'selection.json')
    assert not data['test_read'] and data['provenance']['config_sha256']==sha256(config)
    assert data['gate_passed'],'No geometry family passed the numerical validation gate'
    return data


def final_plan(spec):
    return [(d,s,f) for d in spec['datasets'] for s in spec['seeds'] for f in spec['families']+['raw_frozen']]


def final(spec,config,index):
    selection=frozen_selection(spec,config)
    dataset,seed,family=final_plan(spec)[index]
    root=Path(spec['output_root'])
    folder=root/'final'/dataset/str(seed)/family
    folder.mkdir(parents=True,exist_ok=False)
    record=dict(provenance=provenance(config),dataset=dataset,seed=seed,family=family,
                selection_sha256=sha256(root/'selection.json'),fits=[],failures=[],complete=True)
    if family!='raw_frozen' and not selection['families'][family]['qualified']:
        record.update(skipped=True,reason='Whole-family validation gate failed',test_read=False)
        write_json(folder/'result.json',record)
        return
    y,vy,ids=load_data(spec,config,dataset,seed,spec['primary_train_size'])
    record['subset_sha256']=ids
    device=gpu()
    for arm in ['raw_frozen'] if family=='raw_frozen' else spec['arms']:
        if arm=='raw_frozen':
            model,fit=train_frozen(y,vy,'raw_trans',spec['baseline_candidate'],spec['baseline_training'],seed,device,folder/arm)
            predictor=lambda g:predict_frozen(model,g,device)
        else:
            model,fit=train_moe(y,vy,arm,selection['families'][family]['candidate'],spec['training'],seed,device,folder/arm)
            predictor=lambda g:predict(model,g,device)
        row={k:fit[k] for k in ('train_rmse_r','validation_rmse_r','selected_step')}
        row['arm']=arm
        if not np.isfinite(fit['validation_rmse_r']) or fit['validation_rmse_r']>=spec['validation_gate_rmse_r']:
            row.update(test_read=False,reason='Numerical validation failure')
            record['failures'].append(row)
        else:
            row['metrics']=evaluate(spec,dataset,folder/arm,predictor)
            record['fits'].append(row)
        del model
        torch.cuda.empty_cache()
    write_json(folder/'result.json',record)


def audit(spec,config,index):
    selection=frozen_selection(spec,config)
    dataset=spec['datasets'][index]
    root=Path(spec['output_root'])
    rows,failures=[],[]
    for d,seed,family in final_plan(spec):
        if d!=dataset:
            continue
        folder=root/'final'/dataset/str(seed)/family
        result=read(folder/'result.json')
        assert result['complete'] and result['provenance']['config_sha256']==sha256(config)
        assert result['selection_sha256']==sha256(root/'selection.json')
        assert (result['dataset'],result['seed'],result['family'])==(dataset,seed,family)
        if result.get('skipped'):
            assert family!='raw_frozen' and not selection['families'][family]['qualified']
            assert not result['test_read'] and not list(folder.rglob('*predictions.npz'))
            failures.append(dict(dataset=dataset,seed=seed,family=family,reason=result['reason']))
            continue
        expected=['raw_frozen'] if family=='raw_frozen' else spec['arms']
        assert sorted(r['arm'] for r in result['fits']+result['failures'])==sorted(expected)
        _,_,ids=load_data(spec,config,dataset,seed,spec['primary_train_size'],validation=False)
        assert ids==result['subset_sha256']
        for failure in result['failures']:
            assert not list((folder/failure['arm']).glob('*predictions.npz'))
            failures.append(dict(dataset=dataset,seed=seed,family=family,**failure))
        for fit in result['fits']:
            assert np.isfinite(fit['validation_rmse_r']) and fit['validation_rmse_r']<spec['validation_gate_rmse_r']
            assert set(r['role'] for r in fit['metrics'])=={'test','unseen_scale'}
            for role in ('test','unseen_scale'):
                truth,scales=load_truth(spec,dataset,role)
                with np.load(folder/fit['arm']/f'{role}_predictions.npz') as data:
                    prediction=data['prediction']
                assert prediction.shape==truth.shape and np.isfinite(prediction).all()
                np.testing.assert_allclose(prediction[:,:,0],truth[:,:,0],rtol=0,atol=1e-6)
                records=[r for r in fit['metrics'] if r['role']==role]
                assert sorted(r['scale_id'] for r in records)==[-1]+sorted(np.unique(scales).tolist())
                for metric in records:
                    mask=np.ones(len(truth),bool) if metric['scale_id']==-1 else scales==metric['scale_id']
                    delta=prediction[mask,:,1:].astype(np.float64)-truth[mask,:,1:].astype(np.float64)
                    value=float(np.sqrt(np.sum(delta**2)/(len(delta)*7*31)))
                    np.testing.assert_allclose(value,metric['position_rmse_r'],rtol=1e-10,atol=1e-12)
                    rows.append(dict(dataset=dataset,seed=seed,family=family,arm=fit['arm'],role=role,
                                     scale_id=metric['scale_id'],position_rmse_r=value))
    (root/'audit').mkdir(exist_ok=True)
    write_json(root/'audit'/f'{dataset}.json',dict(provenance=provenance(config),audit_passed=True,rows=rows,failures=failures))


def merge(spec,config,index=0):
    frozen_selection(spec,config)
    root=Path(spec['output_root'])
    rows,failures=[],[]
    for dataset in spec['datasets']:
        data=read(root/'audit'/f'{dataset}.json')
        assert data['audit_passed'] and data['provenance']['config_sha256']==sha256(config)
        rows.extend(data['rows'])
        failures.extend(data['failures'])
    summary=[]
    for family in spec['families']+['raw_frozen']:
        for arm in ['raw_frozen'] if family=='raw_frozen' else spec['arms']:
            for role in ('test','unseen_scale'):
                selected=[r for r in rows if r['family']==family and r['arm']==arm and r['role']==role and r['scale_id']==-1]
                keys=[(r['dataset'],r['seed']) for r in selected]
                assert len(set(keys))==len(keys)
                complete=set(keys)=={(d,s) for d in spec['datasets'] for s in spec['seeds']}
                summary.append(dict(family=family,arm=arm,role=role,complete=complete,evaluated=len(selected),
                    position_rmse_r=float(np.mean([r['position_rmse_r'] for r in selected])) if complete else None))
    for suffix in ('*.pt','*.pth','*.ckpt'):
        assert not list(root.rglob(suffix))
    if rows:
        with (root/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(root/'final_audit.json',dict(provenance=provenance(config),audit_passed=True,summary=summary,failures=failures))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['prepare','fit_check','screen','promote','refine','select','final','audit','merge'])
    p.add_argument('--config',default=DEFAULT)
    p.add_argument('--index',type=int,default=0)
    args=p.parse_args()
    spec=read(args.config)
    root=Path(spec['output_root'])
    assert sha256(args.config)==sha256(root/'config.frozen.json')
    (root/'events').mkdir(exist_ok=True)
    path=root/'events'/f'{args.phase}_{args.index}.json'
    assert not path.exists()
    event=dict(phase=args.phase,index=args.index,status='RUNNING',started=provenance(args.config))
    if args.phase in ('fit_check','screen','refine','final') and torch.cuda.is_available():
        event['gpu']=torch.cuda.get_device_name(0)
    write_json(path,event)
    try:
        globals()[args.phase](spec,args.config,args.index)
        event['status']='COMPLETED'
    except BaseException:
        event.update(status='FAILED',traceback=traceback.format_exc())
        raise
    finally:
        event['ended']=datetime.datetime.now().astimezone().isoformat()
        write_json(path,event)


if __name__=='__main__':
    main()
