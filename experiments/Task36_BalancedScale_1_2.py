"""Task3+6 1.2: final evaluation at two data sizes after validation-only selection."""
import argparse
import csv
import datetime
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256, write_json
from FMT_Utils.Task36Balanced_3D import train_balanced, predict, classification_metrics
from FMT_Utils.Task36MultiGate_3D import task_flags
from FMT_Utils.Task36ScaleData_3D import prepare, prepare_test
from FMT_Utils.Task36Labels_3D import load_labels
from experiments.Task6_DirectNeural_5_1 import read, provenance, load_data, load_truth, gpu

DEFAULT = 'config/Verify_Task36_BalancedScale_1.2.json'


def candidates_for(spec, variant):
    # Multiplying a single task by a constant is not the multi-task balance search.
    return spec['candidates'][:1] if variant.startswith('single_') else spec['candidates']


def search_plan(spec):
    return [(d,n,v,c) for d in spec['datasets'] for n in spec['train_sizes']
            for v in spec['variants'] for c in candidates_for(spec,v)]


def final_plan(spec):
    return [(d,n,s,v) for d in spec['datasets'] for n in spec['train_sizes']
            for s in spec['seeds'] for v in spec['variants']]


def train_data(spec,config,dataset,seed,n):
    y,vy,ids=load_data(spec,config,dataset,seed,n)
    labels=load_labels(spec,config,dataset,f'train_{seed}')[:n]
    vlabels=load_labels(spec,config,dataset,'validation')
    assert len(labels)==len(y) and len(vlabels)==len(vy)
    return y,labels,vy,vlabels,ids


def search(spec,config,index):
    dataset,n,variant,candidate=search_plan(spec)[index]
    folder=Path(spec['output_root'])/'search'/dataset/str(n)/variant/candidate['id']
    y,labels,vy,vlabels,ids=train_data(spec,config,dataset,spec['search_seed'],n)
    model,fit=train_balanced(y,labels,vy,vlabels,variant,candidate,spec['training'],
                            spec['search_seed'],gpu(),folder)
    write_json(folder/'result.json',dict(provenance=provenance(config),dataset=dataset,train_size=n,
        variant=variant,candidate=candidate,subset_sha256=ids,validation=fit['validation'],
        train=fit['train'],test_read=False,selected_step=fit['selected_step'],parameters=fit['parameters']))
    del model


def select(spec,config,index=0):
    root=Path(spec['output_root'])
    assert not (root/'selection.json').exists()
    rows=[]
    for dataset,n,variant,candidate in search_plan(spec):
        row=read(root/'search'/dataset/str(n)/variant/candidate['id']/'result.json')
        assert row['provenance']['config_sha256']==sha256(config) and not row['test_read']
        assert (row['dataset'],row['train_size'],row['variant'],row['candidate'])==(dataset,n,variant,candidate)
        assert np.isfinite(row['validation']['selection_score'])
        rows.append(row)
    choices,scores={},{}
    for n in spec['train_sizes']:
        choices[str(n)],scores[str(n)]={},{}
        for variant in spec['variants']:
            values={}
            for candidate in candidates_for(spec,variant):
                chosen=[r for r in rows if (r['train_size'],r['variant'],r['candidate']['id'])==(n,variant,candidate['id'])]
                assert len(chosen)==len(spec['datasets'])
                values[candidate['id']]=float(np.mean([r['validation']['selection_score'] for r in chosen]))
            best=min(candidates_for(spec,variant),key=lambda c:(values[c['id']],c['id']))
            choices[str(n)][variant]=best
            scores[str(n)][variant]=values
    warnings=[r for r in rows if r['validation']['task6_rmse_r'] is not None and
              r['validation']['task6_rmse_r']>=spec['validation_gate_rmse_r']]
    write_json(root/'selection.json',dict(provenance=provenance(config),rows=rows,choices=choices,scores=scores,
        engineering_warnings=warnings,test_read=False,rule=spec['selection'],evaluation_policy=spec['evaluation_policy']))
    shutil.copyfile(root/'selection.json',root/'selection.before_test.json')


def frozen_selection(spec,config):
    root=Path(spec['output_root'])
    assert sha256(root/'selection.json')==sha256(root/'selection.before_test.json')
    data=read(root/'selection.json')
    assert data['provenance']['config_sha256']==sha256(config) and not data['test_read']
    for n in spec['train_sizes']:
        assert set(data['choices'][str(n)])==set(spec['variants'])
        for variant in spec['variants']:
            assert data['choices'][str(n)][variant] in candidates_for(spec,variant)
    return data


def evaluate(spec,config,dataset,variant,seed,model,fit,folder,device):
    rows=[]
    use3,use6=task_flags(variant)
    for role in ('test','unseen_scale'):
        truth,scales=load_truth(spec,dataset,role)
        labels=load_labels(spec,config,dataset,role)
        probability,prediction=predict(model,truth,device)
        arrays={}
        if use3:
            arrays['probability']=probability
        if use6:
            arrays['prediction']=prediction
        np.savez(folder/f'{role}_predictions.npz',**arrays)
        for scale in [-1]+np.unique(scales).tolist():
            mask=np.ones(len(truth),bool) if scale==-1 else scales==scale
            delta=prediction[mask,:,1:].astype(np.float64)-truth[mask,:,1:].astype(np.float64)
            row=dict(dataset=dataset,seed=seed,variant=variant,role=role,scale_id=int(scale),samples=int(mask.sum()),
                task6_rmse_r=float(np.sqrt(np.sum(delta**2)/(mask.sum()*7*31))) if use6 else None)
            row.update(classification_metrics(labels[mask],probability[mask],fit['threshold']) if use3 else
                       dict(average_precision=None,auroc=None,f1=None,positive_fraction=None,threshold=None))
            rows.append(row)
    return rows


def final(spec,config,index):
    selection=frozen_selection(spec,config)
    dataset,n,seed,variant=final_plan(spec)[index]
    candidate=selection['choices'][str(n)][variant]
    root=Path(spec['output_root'])
    folder=root/'final'/dataset/str(n)/str(seed)/variant
    y,labels,vy,vlabels,ids=train_data(spec,config,dataset,seed,n)
    device=gpu()
    model,fit=train_balanced(y,labels,vy,vlabels,variant,candidate,spec['training'],seed,device,folder)
    error=fit['validation']['task6_rmse_r']
    assert error is None or np.isfinite(error)
    failed=error is not None and error>=spec['validation_gate_rmse_r']
    rows=evaluate(spec,config,dataset,variant,seed,model,fit,folder,device)
    for row in rows:
        row.update(train_size=n,candidate=candidate['id'],engineering_failed=failed)
    write_json(folder/'result.json',dict(provenance=provenance(config),dataset=dataset,train_size=n,seed=seed,
        variant=variant,candidate=candidate,subset_sha256=ids,selection_sha256=sha256(root/'selection.json'),
        complete=True,engineering_failed=failed,validation=fit['validation'],test_read=True,
        metrics=rows,threshold=fit['threshold']))


def audit(spec,config,index):
    frozen_selection(spec,config)
    root=Path(spec['output_root'])
    dataset,n=[(d,n) for d in spec['datasets'] for n in spec['train_sizes']][index]
    rows,failures=[],[]
    for d,size,seed,variant in final_plan(spec):
        if d!=dataset or size!=n:
            continue
        folder=root/'final'/dataset/str(n)/str(seed)/variant
        result=read(folder/'result.json')
        assert result['complete'] and result['provenance']['config_sha256']==sha256(config)
        assert (result['dataset'],result['seed'],result['variant'])==(dataset,seed,variant)
        assert result['selection_sha256']==sha256(root/'selection.json')
        _,_,ids=load_data(spec,config,dataset,seed,n,validation=False)
        assert result['subset_sha256']==ids
        assert result['test_read'] and result['train_size']==n
        assert result['candidate']==frozen_selection(spec,config)['choices'][str(n)][variant]
        assert result['engineering_failed']==(result['validation']['task6_rmse_r'] is not None and result['validation']['task6_rmse_r']>=spec['validation_gate_rmse_r'])
        if result['engineering_failed']:
            failures.append(dict(dataset=dataset,train_size=n,seed=seed,variant=variant,validation=result['validation']))
        use3,use6=task_flags(variant)
        for role in ('test','unseen_scale'):
            truth,scales=load_truth(spec,dataset,role)
            labels=load_labels(spec,config,dataset,role)
            with np.load(folder/f'{role}_predictions.npz') as data:
                probability=data['probability'].copy() if use3 else None
                prediction=data['prediction'].copy() if use6 else None
            if use3:
                assert probability.shape==(len(truth),) and np.isfinite(probability).all()
                assert ((probability>=0)&(probability<=1)).all()
            if use6:
                assert prediction.shape==truth.shape and np.isfinite(prediction).all()
                np.testing.assert_allclose(prediction[:,:,0],truth[:,:,0],rtol=0,atol=2e-5)
            records=[r for r in result['metrics'] if r['role']==role]
            assert sorted(r['scale_id'] for r in records)==[-1]+sorted(np.unique(scales).tolist())
            for row in records:
                assert row['train_size']==n and row['candidate']==result['candidate']['id']
                mask=np.ones(len(truth),bool) if row['scale_id']==-1 else scales==row['scale_id']
                if use6:
                    diff=prediction[mask,:,1:].astype(np.float64)-truth[mask,:,1:].astype(np.float64)
                    error=float(np.sqrt(np.sum(diff**2)/(mask.sum()*7*31)))
                    np.testing.assert_allclose(error,row['task6_rmse_r'],rtol=1e-12,atol=1e-12)
                if use3:
                    fresh=classification_metrics(labels[mask],probability[mask],result['threshold'])
                    for key,value in fresh.items():
                        if value is None:
                            assert row[key] is None
                        else:
                            np.testing.assert_allclose(value,row[key],rtol=1e-12,atol=1e-12)
                    truth_labels=labels[mask].astype(bool)
                    predicted=probability[mask]>=result['threshold']
                    tp=int((truth_labels&predicted).sum())
                    denominator=int(truth_labels.sum()+predicted.sum())
                    np.testing.assert_allclose(row['f1'],2*tp/denominator if denominator else 0)
                rows.append(row)
    (root/'audit').mkdir(exist_ok=True)
    write_json(root/'audit'/f'{dataset}_{n}.json',dict(provenance=provenance(config),audit_passed=True,rows=rows,failures=failures))


def merge(spec,config,index=0):
    frozen_selection(spec,config)
    root=Path(spec['output_root'])
    rows,failures=[],[]
    for dataset in spec['datasets']:
        for n in spec['train_sizes']:
            data=read(root/'audit'/f'{dataset}_{n}.json')
            assert data['audit_passed'] and data['provenance']['config_sha256']==sha256(config)
            rows.extend(data['rows'])
            failures.extend(data['failures'])
    summaries=[]
    expected={(d,s) for d in spec['datasets'] for s in spec['seeds']}
    for n in spec['train_sizes']:
        for variant in spec['variants']:
            for role in ('test','unseen_scale'):
                chosen=[r for r in rows if (r['train_size'],r['variant'],r['role'],r['scale_id'])==(n,variant,role,-1)]
                ids={(r['dataset'],r['seed']) for r in chosen}
                assert len(ids)==len(chosen) and ids==expected
                metrics={key:float(np.mean([r[key] for r in chosen])) if all(r[key] is not None for r in chosen) else None
                         for key in ('average_precision','auroc','f1','task6_rmse_r')}
                summaries.append(dict(train_size=n,variant=variant,role=role,complete=True,evaluated=len(chosen),
                    engineering_failed=sum(r['engineering_failed'] for r in chosen),**metrics))
    for suffix in ('*.pt','*.pth','*.ckpt'):
        assert not list(root.rglob(suffix))
    with (root/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root/'final_audit.json',dict(provenance=provenance(config),audit_passed=True,
        summary=summaries,engineering_failures=failures,all_failures_retained=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','search','select','prepare_test','final','audit','merge'])
    parser.add_argument('--config',default=DEFAULT)
    parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args()
    spec=read(args.config)
    root=Path(spec['output_root'])
    assert sha256(args.config)==sha256(root/'config.frozen.json')
    (root/'events').mkdir(exist_ok=True)
    path=root/'events'/f'{args.phase}_{args.index}.json'
    assert not path.exists()
    event=dict(phase=args.phase,index=args.index,status='RUNNING',started=provenance(args.config))
    if args.phase in ('search','final') and torch.cuda.is_available():
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
