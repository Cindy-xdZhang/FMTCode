"""Joint classification/reconstruction experiment with frozen splits and controls."""
import argparse
import csv
import datetime
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256,write_json
from FMT_Utils.Task36MultiGate_3D import train_joint,predict,classification_metrics,task_flags
from FMT_Utils.Task36Labels_3D import prepare,prepare_test,load_labels
from experiments.Task6_DirectNeural_5_1 import read,provenance,load_data,load_truth,gpu

DEFAULT='config/Verify_Task36_MultiGate_1.1.json'


def train_data(spec,config,dataset,seed):
    y,vy,ids=load_data(spec,config,dataset,seed,spec['primary_train_size'])
    labels=load_labels(spec,config,dataset,f'train_{seed}')
    vlabels=load_labels(spec,config,dataset,'validation')
    assert len(labels)==len(y) and len(vlabels)==len(vy)
    return y,labels,vy,vlabels,ids


def search(spec,config,index):
    dataset,variant=[(d,v) for d in spec['datasets'] for v in spec['variants']][index]
    folder=Path(spec['output_root'])/'search'/dataset/variant
    y,labels,vy,vlabels,ids=train_data(spec,config,dataset,spec['search_seed'])
    model,fit=train_joint(y,labels,vy,vlabels,variant,spec['candidate'],spec['training'],
                        spec['search_seed'],gpu(),folder)
    write_json(folder/'result.json',dict(provenance=provenance(config),dataset=dataset,variant=variant,
        subset_sha256=ids,validation=fit['validation'],train=fit['train'],test_read=False,
        selected_step=fit['selected_step'],parameters=fit['parameters']))
    del model


def select(spec,config,index=0):
    root=Path(spec['output_root'])
    assert not (root/'selection.json').exists()
    rows=[]
    for dataset in spec['datasets']:
        for variant in spec['variants']:
            row=read(root/'search'/dataset/variant/'result.json')
            assert row['provenance']['config_sha256']==sha256(config) and not row['test_read']
            assert (row['dataset'],row['variant'])==(dataset,variant)
            rows.append(row)
    failures=[]
    for row in rows:
        value=row['validation']['task6_rmse_r']
        if value is not None and (not np.isfinite(value) or value>=spec['validation_gate_rmse_r']):
            failures.append(row)
    write_json(root/'selection.json',dict(provenance=provenance(config),rows=rows,failures=failures,
        gate_passed=not failures,test_read=False,candidate=spec['candidate'],
        rule='All seven preregistered variants retained; no test-based method selection'))
    shutil.copyfile(root/'selection.json',root/'selection.before_test.json')


def frozen_selection(spec,config):
    root=Path(spec['output_root'])
    assert sha256(root/'selection.json')==sha256(root/'selection.before_test.json')
    data=read(root/'selection.json')
    assert data['gate_passed'] and not data['test_read']
    assert data['provenance']['config_sha256']==sha256(config) and data['candidate']==spec['candidate']
    return data


def final_plan(spec):
    return [(d,s,v) for d in spec['datasets'] for s in spec['seeds'] for v in spec['variants']]


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
    frozen_selection(spec,config)
    dataset,seed,variant=final_plan(spec)[index]
    root=Path(spec['output_root'])
    folder=root/'final'/dataset/str(seed)/variant
    y,labels,vy,vlabels,ids=train_data(spec,config,dataset,seed)
    device=gpu()
    model,fit=train_joint(y,labels,vy,vlabels,variant,spec['candidate'],spec['training'],seed,device,folder)
    error=fit['validation']['task6_rmse_r']
    failed=error is not None and (not np.isfinite(error) or error>=spec['validation_gate_rmse_r'])
    rows=[] if failed else evaluate(spec,config,dataset,variant,seed,model,fit,folder,device)
    write_json(folder/'result.json',dict(provenance=provenance(config),dataset=dataset,seed=seed,variant=variant,
        subset_sha256=ids,selection_sha256=sha256(root/'selection.json'),complete=True,failed=failed,
        validation=fit['validation'],test_read=not failed,metrics=rows,threshold=fit['threshold']))


def audit(spec,config,index):
    frozen_selection(spec,config)
    root=Path(spec['output_root'])
    dataset=spec['datasets'][index]
    rows,failures=[],[]
    for d,seed,variant in final_plan(spec):
        if d!=dataset:
            continue
        folder=root/'final'/dataset/str(seed)/variant
        result=read(folder/'result.json')
        assert result['complete'] and result['provenance']['config_sha256']==sha256(config)
        assert (result['dataset'],result['seed'],result['variant'])==(dataset,seed,variant)
        assert result['selection_sha256']==sha256(root/'selection.json')
        _,_,ids=load_data(spec,config,dataset,seed,spec['primary_train_size'],validation=False)
        assert result['subset_sha256']==ids
        if result['failed']:
            assert not result['test_read'] and not list(folder.glob('*predictions.npz'))
            failures.append(result)
            continue
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
    summaries=[]
    for variant in spec['variants']:
        for role in ('test','unseen_scale'):
            chosen=[r for r in rows if (r['variant'],r['role'],r['scale_id'])==(variant,role,-1)]
            ids={(r['dataset'],r['seed']) for r in chosen}
            assert len(ids)==len(chosen)
            complete=ids=={(d,s) for d in spec['datasets'] for s in spec['seeds']}
            metrics={key:float(np.mean([r[key] for r in chosen])) if complete and all(r[key] is not None for r in chosen) else None
                     for key in ('average_precision','auroc','f1','task6_rmse_r')}
            summaries.append(dict(variant=variant,role=role,complete=complete,evaluated=len(chosen),**metrics))
    for suffix in ('*.pt','*.pth','*.ckpt'):
        assert not list(root.rglob(suffix))
    if rows:
        with (root/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(root/'final_audit.json',dict(provenance=provenance(config),audit_passed=True,summary=summaries,failures=failures))


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
