"""Screen larger regularized networks, then compare both arms on nine frozen flows."""
import argparse
import datetime
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from FMT_Utils.FlowMapData_3D import sha256, write_json
from FMT_Utils.Task6PNNTuning_3D import train_tuned, predict
from FMT_Utils.Task6PNNTrans_3D import train_model as train_frozen
from experiments.Task6_DirectNeural_5_1 import read, provenance, prepare as prepare_flow, load_data, gpu
from experiments.Task6_PNNTrans_1_1 import frozen_selection, evaluate, audit, merge

DEFAULT = 'config/Verify_Task6_PNNTrans_1.2.json'


def baseline_rows(spec):
    root = Path(spec['baseline_cache'])
    assert sha256(root/'config.frozen.json') == spec['baseline_config_sha256']
    assert sha256(root/'selection.json') == spec['baseline_selection_sha256']
    assert sha256(root/'selection.before_test.json') == spec['baseline_selection_sha256']
    selection = read(root/'selection.json')
    assert not selection['test_read']
    rows = [r for r in selection['rows'] if r['candidate'] == spec['baseline_candidate']['id'] and r['arm'] == 'raw_trans']
    assert sorted(r['dataset'] for r in rows) == sorted(spec['datasets'])
    return rows


def prepare(spec, config, index=0):
    reference = baseline_rows(spec)
    for i in range(len(spec['datasets'])):
        prepare_flow(spec,config,i)
    write_json(Path(spec['output_root'])/'baseline_reference.json',dict(provenance=provenance(config),
        rows=reference,source_config_sha256=spec['baseline_config_sha256'],
        selection_sha256=spec['baseline_selection_sha256'],test_read=False))


def capacity_check(spec,config,index):
    arm = ('pnn_trans','raw_trans')[index]
    candidate = next(c for c in spec['candidates'] if c['id']=='c14')
    device = gpu()
    y,_,subset_hash = load_data(spec,config,spec['datasets'][0],spec['search_seed'],128,validation=False)
    training = dict(spec['training'],updates=8,probe_every=4,warmup_updates=2)
    model,result = train_tuned(y,y[:32],arm,candidate,training,spec['search_seed'],device,
                              Path(spec['output_root'])/'capacity_check'/arm)
    write_json(Path(spec['output_root'])/'capacity_check'/f'{arm}.json',dict(provenance=provenance(config),
        device=torch.cuda.get_device_name(0),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        finite=True,subset_sha256=subset_hash,structure=result['structure'],test_read=False,
        purpose='Largest network batch128 memory/backprop check; not a generalization or convergence result'))
    del model


def screen(spec,config,index):
    dataset,candidate = [(d,c) for d in spec['screen_datasets'] for c in spec['candidates']][index]
    device = gpu()
    y,vy,subset_hash = load_data(spec,config,dataset,spec['search_seed'],spec['primary_train_size'])
    folder = Path(spec['output_root'])/'screen'/dataset/candidate['id']
    model,result = train_tuned(y,vy,'pnn_trans',candidate,spec['screening'],spec['search_seed'],device,folder/'pnn_trans')
    write_json(folder/'result.json',dict(provenance=provenance(config),dataset=dataset,candidate=candidate['id'],
        device=torch.cuda.get_device_name(0),subset_sha256=subset_hash,arm='pnn_trans',
        train_rmse_r=result['train_rmse_r'],validation_rmse_r=result['validation_rmse_r'],
        selected_step=result['selected_step'],test_read=False))
    del model


def promotion_ranking(rows,baseline,spec):
    reference = {r['dataset']:r['validation_rmse_r'] for r in baseline}
    scores = {}
    for c in spec['candidates']:
        values = [r['validation_rmse_r']/reference[r['dataset']] for r in rows if r['candidate']==c['id']]
        assert len(values)==len(spec['screen_datasets']) and np.isfinite(values).all()
        scores[c['id']] = float(np.mean(values))
    selected = sorted(spec['candidates'],key=lambda c:(scores[c['id']],c['id']))[:spec['promote_count']]
    return scores,selected


def promote(spec,config,index=0):
    root = Path(spec['output_root'])
    assert not (root/'promotion.json').exists()
    rows = []
    for d in spec['screen_datasets']:
        for c in spec['candidates']:
            row = read(root/'screen'/d/c['id']/'result.json')
            assert row['provenance']['config_sha256']==sha256(config) and not row['test_read']
            rows.append(row)
    scores,chosen = promotion_ranking(rows,baseline_rows(spec),spec)
    write_json(root/'promotion.json',dict(provenance=provenance(config),scores=scores,candidates=chosen,
        rows=rows,test_read=False,rule='Equal-flow mean ratio to frozen Raw on preregistered three flows'))
    shutil.copyfile(root/'promotion.json',root/'promotion.frozen.json')


def promoted(spec,config):
    root = Path(spec['output_root'])
    assert sha256(root/'promotion.json')==sha256(root/'promotion.frozen.json')
    data = read(root/'promotion.json')
    assert not data['test_read'] and data['provenance']['config_sha256']==sha256(config)
    return data['candidates']


def refine(spec,config,index):
    candidates = promoted(spec,config)
    dataset,candidate = [(d,c) for d in spec['datasets'] for c in candidates][index]
    root = Path(spec['output_root'])
    folder = root/'refine'/dataset/candidate['id']
    device = gpu()
    y,vy,subset_hash = load_data(spec,config,dataset,spec['search_seed'],spec['primary_train_size'])
    rows=[]
    for arm in ('pnn_trans','raw_trans'):
        model,result = train_tuned(y,vy,arm,candidate,spec['training'],spec['search_seed'],device,folder/arm)
        rows.append(dict(arm=arm,candidate=candidate['id'],train_rmse_r=result['train_rmse_r'],
            validation_rmse_r=result['validation_rmse_r'],selected_step=result['selected_step']))
        del model
        torch.cuda.empty_cache()
    write_json(folder/'result.json',dict(provenance=provenance(config),dataset=dataset,rows=rows,
        device=torch.cuda.get_device_name(0),subset_sha256=subset_hash,test_read=False,
        promotion_sha256=sha256(root/'promotion.json')))


def selection_result(rows,baseline,candidates,spec):
    scores={c['id']:float(np.mean([r['validation_rmse_r'] for r in rows
        if r['candidate']==c['id'] and r['arm']=='pnn_trans'])) for c in candidates}
    assert np.isfinite(list(scores.values())).all()
    chosen=min(candidates,key=lambda c:(scores[c['id']],c['id']))
    matched=[r for r in rows if r['candidate']==chosen['id']]
    assert len(matched)==len(spec['datasets'])*2
    means={arm:float(np.mean([r['validation_rmse_r'] for r in matched if r['arm']==arm]))
           for arm in ('pnn_trans','raw_trans')}
    means['raw_frozen']=float(np.mean([r['validation_rmse_r'] for r in baseline]))
    failures=[r for r in matched if r['validation_rmse_r']>=spec['validation_gate_rmse_r']]
    beats=means['pnn_trans']<min(means['raw_trans'],means['raw_frozen'])
    return dict(candidate=chosen,scores=scores,rows=rows,baseline_rows=baseline,means=means,
        numeric_gate_passed=not failures,beats_both_baselines=beats,gate_passed=not failures and beats,
        failures=failures,test_read=False,rule=spec['selection'])


def select(spec,config,index=0):
    root=Path(spec['output_root'])
    assert not (root/'selection.json').exists()
    candidates=promoted(spec,config)
    rows=[]
    for d in spec['datasets']:
        for c in candidates:
            result=read(root/'refine'/d/c['id']/'result.json')
            assert not result['test_read'] and result['provenance']['config_sha256']==sha256(config)
            assert result['promotion_sha256']==sha256(root/'promotion.json')
            assert sorted(r['arm'] for r in result['rows'])==['pnn_trans','raw_trans']
            rows.extend(dict(dataset=d,**r) for r in result['rows'])
    result=selection_result(rows,baseline_rows(spec),candidates,spec)
    result.update(provenance=provenance(config),promotion_sha256=sha256(root/'promotion.json'))
    write_json(root/'selection.json',result)
    shutil.copyfile(root/'selection.json',root/'selection.before_test.json')


def final(spec,config,index):
    selection=frozen_selection(spec,config)
    dataset,n,seed=[(d,n,s) for d in spec['datasets'] for n in spec['train_sizes'] for s in spec['seeds']][index]
    root=Path(spec['output_root'])
    folder=root/'final'/dataset/str(n)/str(seed)
    folder.mkdir(parents=True,exist_ok=False)
    device=gpu()
    y,vy,subset_hash=load_data(spec,config,dataset,seed,n)
    fits,failures=[],[]
    for arm in spec['arms']:
        if arm=='raw_frozen':
            model,result=train_frozen(y,vy,'raw_trans',spec['baseline_candidate'],spec['baseline_training'],seed,device,folder/arm)
        else:
            model,result=train_tuned(y,vy,arm,selection['candidate'],spec['training'],seed,device,folder/arm)
        row=dict(arm=arm,train_rmse_r=result['train_rmse_r'],validation_rmse_r=result['validation_rmse_r'],selected_step=result['selected_step'])
        if result['validation_rmse_r']>=spec['validation_gate_rmse_r']:
            row.update(test_read=False,reason='Validation error exceeds threshold; no test for this model')
            failures.append(row)
        else:
            row['metrics']=evaluate(spec,dataset,folder/arm,lambda geometry:predict(model,geometry,device))
            fits.append(row)
        del model
        torch.cuda.empty_cache()
    write_json(folder/'result.json',dict(provenance=provenance(config),dataset=dataset,train_size=n,seed=seed,
        device=torch.cuda.get_device_name(0),subset_sha256=subset_hash,selection_sha256=sha256(root/'selection.json'),
        fits=fits,failures=failures,complete=True,passed=not failures,model_files='none'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['prepare','capacity_check','screen','promote','refine','select','final','audit','merge'])
    p.add_argument('--config',default=DEFAULT)
    p.add_argument('--index',type=int,default=0)
    args=p.parse_args()
    spec=read(args.config)
    root=Path(spec['output_root'])
    assert sha256(args.config)==sha256(root/'config.frozen.json')
    events=root/'events'
    events.mkdir(exist_ok=True)
    path=events/f'{args.phase}_{args.index}.json'
    assert not path.exists()
    event=dict(phase=args.phase,index=args.index,started=provenance(args.config),status='RUNNING')
    if args.phase in ('capacity_check','screen','refine','final') and torch.cuda.is_available():
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
