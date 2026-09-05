"""Preflight, fit clean controls, and evaluate one paired corruption grid.

One runner serves clean and robustness experiment IDs. No test metrics are
computed until every clean model and threshold in a shard has been frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, jaccard_score
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, average_precision_score

from FMT_Utils.GeometricControls_3D import (
    validate_cache, representations, perturb, pad_auxiliary, array_hash,
)
from FMT_Utils.Task12Evaluation_3D import fit_kmeans_transform, calibrate_vortex_cluster
from FMT_Utils.PathlineClassifier_3D import PathlineFMTResidualClassifier3D, residual_model_kwargs
from experiments.Verify_Task3_FMTClassifier import (
    _classification_metrics, _select_f1_threshold, _normalize_train_only, _loader, _predict,
)
from experiments.Verify_Task3_FMTResidual import (
    _train_one, _load_raw_model, _predict_components, _probabilities,
)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        json.dump(data,f,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else
                  x.item() if isinstance(x,np.generic) else str(x))


def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(row.keys() for row in rows))))
        w.writeheader(); w.writerows(rows)


def sources(spec, task, dataset, phase):
    group='new2' if dataset in {'boeing747','smokeBuoyancy'} else 'old8'
    root=Path(spec['roots']['multiscale' if task=='Task5' else 'fixed'])
    out=root/'outputs'
    if task=='Task5':
        base=out/'mainExp_Task5_3D_1.1'
        return (base/f'{phase}_cache'/dataset,
                base/f'{phase}_labels_{group}'/'labels'/dataset,
                base/f'development_{group}'/'baselines'/'checkpoints', 6 if phase=='development' else 4)
    cache=(out/'mainExp_Task123NewFlows_1.1'/'development_cache' if group=='new2' else
           out/'Verify_Task2Universality_1.1'/'cache')
    if task=='Task1' and phase=='confirmation':
        cache=(out/'mainExp_Task123NewFlows_1.1'/'confirmation_cache' if group=='new2' else
               out/'mainExp_Task3Universality_2.2'/'confirmation_cache')
    labels=out/'mainExp_Task3_3D_3.2_global_ivd'/f'development_labels_{group}'/'labels'/dataset
    backbone=out/'mainExp_Task3_3D_3.2_global_ivd'/f'development_{group}'/'baselines'/'checkpoints'
    return cache/dataset, None if task=='Task1' else labels, backbone, 4 if phase=='confirmation' and task=='Task1' else 10


def load_records(spec, task, dataset, phase, ordinals):
    cache, labels, _, expected=sources(spec,task,dataset,phase)
    paths=sorted(cache.glob('slice_*.npz'))
    if len(paths)!=expected: raise ValueError(f'{cache}: expected {expected}, got {len(paths)}')
    records=[]
    for ordinal in ordinals:
        path=paths[ordinal]
        with np.load(path,allow_pickle=False) as z: record=validate_cache(z)
        if labels is not None:
            label_path=labels/path.name
            with np.load(label_path,allow_pickle=False) as z:
                target=np.asarray(z['labels'],dtype=np.float32)
                meta=json.loads(str(z['metadata_json']))
            percentile=meta.get('label_value',meta.get('ivd_percentile'))
            if percentile is None or float(percentile)!=95:
                raise ValueError(f'{label_path}: label definition is not p95')
            named=str(meta.get('source_cache','')).replace('\\','/').split('/')[-1]
            if named!=path.name: raise ValueError(f'{label_path}: source identity mismatch')
            if not np.array_equal(target,record['labels']):
                raise ValueError(f'{label_path}: labels differ from frozen source reference')
            record['labels']=target
        if record['labels'].shape!=(len(record['raw']),) or not np.isin(record['labels'],[0,1]).all():
            raise ValueError('label rows do not match retained primitives')
        record.update(path=str(path),context=str(path),ordinal=ordinal)
        records.append(record)
    return records


def all_splits(spec,task,dataset,confirmation=False):
    split=spec[task.lower()]
    if confirmation:
        return load_records(spec,task,dataset,'development' if task=='Task3' else 'confirmation',split['confirmation'])
    return (load_records(spec,task,dataset,'development',split['train']),
            load_records(spec,task,dataset,'development',split['validation']))


def preflight(spec, config_path):
    output=Path(spec['output_root'])/'preflight.json'
    if output.exists(): raise FileExistsError('preflight already exists; do not overwrite evidence')
    certificates=[]
    for task in spec['tasks']:
        for dataset in spec['datasets']:
            train,val=all_splits(spec,task,dataset)
            test=all_splits(spec,task,dataset,True)
            groups=[train,val,test]
            file_sets=[{r['path'] for r in rows} for rows in groups]
            if any(file_sets[a]&file_sets[b] for a,b in ((0,1),(0,2),(1,2))):
                raise ValueError('train/validation/confirmation source files overlap')
            for rows,role in zip(groups,('train','validation','confirmation')):
                for r in rows:
                    certificates.append({'task':task,'dataset':dataset,'role':role,
                                         'path':r['path'],'source_sha256':sha(r['path']),
                                         'identity':r['identity'],**r['certificate']})
            if task!='Task1':
                _,_,backbones,_=sources(spec,task,dataset,'development')
                for seed in spec['seeds']:
                    if not (backbones/f'{dataset}_raw_seed{seed}.pt').is_file():
                        raise FileNotFoundError(f'{backbones}/{dataset}_raw_seed{seed}.pt')
                raw_model,_=_load_raw_model(backbones/f'{dataset}_raw_seed{spec["seeds"][0]}.pt',spec['auxiliary_input_width'],'cpu')
                model=PathlineFMTResidualClassifier3D(raw_model,fmt_dim=spec['auxiliary_input_width'],
                    **residual_model_kwargs(spec['model']))
                count=sum(p.numel() for p in model.parameters())
                if count>=148225: raise ValueError(f'capacity guard: {count} >= Raw-wide 148225')
                del model,raw_model
            # Only training features are opened for implementation smoke tests.
            features=representations(train[0],task,device='cpu')
            for values in features.values(): pad_auxiliary(values,spec['auxiliary_input_width'])
            print(f'preflight {task}/{dataset}: valid, feature widths {[v.shape[1] for v in features.values()]}',flush=True)
    write_json(output,{'status':'PASS','config_sha256':sha(config_path),'certificates':certificates,
                       'confirmation_metrics_computed':False,'node':socket.gethostname()})


def feature_split(records,task,device,condition=None,repeat=0):
    raw_parts=[]; feature_parts={}; fingerprints=[]
    for record in records:
        raw,times=(record['raw'],record['times']) if condition is None else perturb(record,condition,repeat)
        features=representations(record,task,raw,times,device)
        raw_parts.append(raw)
        for name,value in features.items(): feature_parts.setdefault(name,[]).append(value)
        fingerprints.append({'identity':record['identity'],'coordinates':array_hash(raw),'times':array_hash(times)})
    return (np.concatenate(raw_parts),{k:np.concatenate(v) for k,v in feature_parts.items()},
            np.concatenate([r['labels'] for r in records]),fingerprints)


def metrics(labels,scores,threshold):
    predicted=scores>=threshold
    if len(np.unique(labels))==2:
        result=_classification_metrics(labels,scores,threshold)
    else:
        result={'f1':float(f1_score(labels,predicted,zero_division=0)),
                'precision':float(precision_score(labels,predicted,zero_division=0)),
                'recall':float(recall_score(labels,predicted,zero_division=0)),
                'balanced_accuracy':float(balanced_accuracy_score(labels,predicted)),
                'average_precision':float(average_precision_score(labels,scores)) if np.any(labels) else 0.0,
                'roc_auc':float('nan'),'predicted_positive_fraction':float(predicted.mean())}
    result.update(iou=float(jaccard_score(labels,predicted,zero_division=0)),
                  ari=float(adjusted_rand_score(labels,predicted)),
                  nmi=float(normalized_mutual_info_score(labels,predicted)),sample_count=len(labels))
    return result


def load_residual(path,width,device):
    state=torch.load(path,map_location='cpu',weights_only=False)
    raw,_=_load_raw_model(state['raw_checkpoint'],width,device)
    model=PathlineFMTResidualClassifier3D(raw,fmt_dim=width,
        **residual_model_kwargs(state['config']['model'])).to(device)
    merged=model.state_dict(); merged.update(state['residual_state_dict']); model.load_state_dict(merged)
    return model.eval(),state


def run_shard(spec,config_path,index):
    jobs=[(task,dataset,int(seed)) for task in spec['tasks'] for dataset in spec['datasets'] for seed in spec['seeds']]
    task,dataset,seed=jobs[index]
    root=Path(spec['output_root']); cert=json.loads((root/'preflight.json').read_text())
    if cert['status']!='PASS' or cert['config_sha256']!=sha(config_path): raise ValueError('missing matching preflight')
    target=root/'shards'/task/dataset/f'seed{seed}'
    if target.exists(): raise FileExistsError(f'refusing to retrain/re-evaluate existing shard {target}')
    target.mkdir(parents=True)
    write_json(target/'started.json',{'config_sha256':sha(config_path),'job':os.environ.get('SLURM_JOB_ID'),
               'array_task':os.environ.get('SLURM_ARRAY_TASK_ID'),'node':socket.gethostname(),'start_time':time.time()})
    device=torch.device('cuda' if task!='Task1' else 'cpu')
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('supervised shard requires allocated GPU')
    train_records,val_records=all_splits(spec,task,dataset)
    for r in train_records+val_records:
        expected=next(c for c in cert['certificates'] if c['task']==task and c['path']==r['path'])
        if sha(r['path'])!=expected['source_sha256']: raise ValueError('cache changed after preflight')
    tr_raw,tr_features,tr_y,_=feature_split(train_records,task,device)
    va_raw,va_features,va_y,_=feature_split(val_records,task,device)
    cutoff=_select_f1_threshold(va_y,va_features['geometry_ivd'][:,0])
    frozen={'geometry_threshold':{'threshold':cutoff,'selection':'validation_F1'}}
    models={}; model_files=[]
    if task=='Task1':
        kmseed=spec['task1']['kmeans_seeds'][spec['seeds'].index(seed)]
        for arm in spec['arms']:
            estimator=fit_kmeans_transform(tr_features[arm],spec['task1']['pca_dims'][arm],kmseed,spec['task1']['kmeans_n_init'])
            cluster=calibrate_vortex_cluster(va_y,estimator.predict(va_features[arm]))
            models[arm]=(estimator,cluster)
            frozen[arm]={'vortex_cluster':cluster,'kmeans_seed':kmseed,'pca_dim':spec['task1']['pca_dims'][arm]}
    else:
        _,_,backbone_dir,_=sources(spec,task,dataset,'development')
        backbone_path=backbone_dir/f'{dataset}_raw_seed{seed}.pt'
        backbone=torch.load(backbone_path,map_location='cpu',weights_only=False)
        width=int(spec['auxiliary_input_width'])
        for arm in spec['arms']:
            train=(tr_raw,pad_auxiliary(tr_features[arm],width),tr_y)
            validation=(va_raw,pad_auxiliary(va_features[arm],width),va_y)
            train,validation,_,stats=_normalize_train_only(train,validation,None,raw_stats=backbone['normalization'])
            directory=target/'temporary_training'/arm
            directory.mkdir(parents=True)
            run_spec={'experiment':spec['experiment'],'model':spec['model'],'fusion':spec['fusion'],
                      'training':spec['training'],'auxiliary_source':'fmt','raw_checkpoint_dir':str(backbone_dir),
                      'raw_wide_parameter_count':148225,'raw_backbone_seed':seed,
                      'evaluation':{'test_enabled':False},'split':spec[task.lower()]}
            result=_train_one(run_spec,dataset,seed,(train,validation,None),stats,device,directory)
            model,state=load_residual(result['checkpoint'],width,device)
            models[arm]=(model,state)
            model_files.append(Path(result['checkpoint']))
            frozen[arm]={'threshold':float(state['threshold']),'alpha':float(state['alpha']),
                         'best_epoch':int(state['best_epoch']),'normalization':state['normalization'],
                         'parameter_count':int(state['total_parameter_count']),
                         'trainable_parameter_count':int(state['trainable_residual_parameter_count']),
                         'backbone_sha256':sha(backbone_path)}
        if len({frozen[a]['parameter_count'] for a in spec['arms']})!=1:
            raise ValueError('matched model parameter counts differ')
        raw_model,_=_load_raw_model(backbone_path,width,device)
        frozen['raw_backbone']={'threshold':float(backbone['threshold']),'backbone_sha256':sha(backbone_path)}
    # Durable freeze barrier before confirmation data or performance is opened.
    write_json(target/'frozen_clean_models.json',frozen)
    del train_records,val_records,tr_raw,tr_features,va_raw,va_features
    test_records=all_splits(spec,task,dataset,True)
    for r in test_records:
        expected=next(c for c in cert['certificates'] if c['task']==task and c['path']==r['path'])
        if sha(r['path'])!=expected['source_sha256']: raise ValueError('confirmation cache changed after preflight')
    scale_ids=np.concatenate([r['scale_id'] for r in test_records])
    all_rows=[]; scale_rows=[]; prediction_files=[]
    for condition in spec['corruptions']:
        repeats=[0] if condition['kind']=='clean' else spec['corruption_seeds']
        for repeat in repeats:
            raw,features,labels,fingerprints=feature_split(test_records,task,device,condition,repeat)
            predictions={'geometry_threshold':features['geometry_ivd'][:,0]}
            thresholds={'geometry_threshold':float(cutoff)}
            for arm,(model,extra) in models.items():
                if task=='Task1':
                    # Oriented distance difference supplies a ranking score;
                    # threshold zero is exactly the calibrated two-centre rule.
                    distances=model.model.transform(model.transform(features[arm]))
                    predictions[arm]=distances[:,1-extra]-distances[:,extra]
                    if extra==1:
                        predictions[arm]=predictions[arm].astype(np.float64)
                        predictions[arm][predictions[arm]==0]=np.nextafter(0.0,-1.0)
                    thresholds[arm]=0.0
                else:
                    stats=extra['normalization']
                    x=((raw-np.asarray(stats['raw_mean']))/np.asarray(stats['raw_std'])).astype(np.float32)
                    f=((pad_auxiliary(features[arm],width)-np.asarray(stats['fmt_mean']))/np.asarray(stats['fmt_std'])).astype(np.float32)
                    loader=_loader((x,f,labels),1024,False,seed,True)
                    _,rlogits,alogits=_predict_components(model,loader,device)
                    predictions[arm]=_probabilities(rlogits,alogits,float(extra['alpha']),extra['config']['model'])
                    thresholds[arm]=float(extra['threshold'])
            if task!='Task1':
                rawstats=backbone['normalization']
                x=((raw-np.asarray(rawstats['raw_mean']))/np.asarray(rawstats['raw_std'])).astype(np.float32)
                loader=_loader((x,np.zeros((len(x),width),np.float32),labels),1024,False,seed,True)
                _,predictions['raw_backbone']=_predict(raw_model,loader,device)
                thresholds['raw_backbone']=float(backbone['threshold'])
            arms=list(predictions)
            evidence=target/f"predictions_{condition['id']}_{repeat}.npz"
            np.savez_compressed(evidence,labels=labels,scores=np.stack([predictions[a] for a in arms]),
                                thresholds=np.asarray([thresholds[a] for a in arms]),arms=np.asarray(arms),
                                scale_ids=scale_ids,fingerprints_json=np.asarray(json.dumps(fingerprints)))
            prediction_files.append({'path':str(evidence),'sha256':sha(evidence)})
            for arm in arms:
                row={'task':task,'dataset':dataset,'seed':seed,'arm':arm,'condition':condition['id'],
                     'kind':condition['kind'],'level':condition['level'],'corruption_seed':repeat,
                     'threshold':thresholds[arm],**metrics(labels,predictions[arm],thresholds[arm])}
                all_rows.append(row)
                if task=='Task5':
                    for scale_id in np.unique(scale_ids):
                        mask=scale_ids==scale_id
                        scale_rows.append({**row,'scale_id':int(scale_id),
                            **metrics(labels[mask],predictions[arm][mask],thresholds[arm])})
            print(f"{task}/{dataset}/seed{seed}/{condition['id']}/{repeat}: evaluated all arms",flush=True)
    for rows,keys in ((all_rows,('arm',)),(scale_rows,('arm','scale_id'))):
        clean={tuple(row[k] for k in keys):row for row in rows if row['condition']=='clean'}
        for row in rows:
            baseline=clean[tuple(row[k] for k in keys)]
            for metric in ('f1','average_precision','iou'):
                row[metric+'_drop_from_clean']=float(baseline[metric]-row[metric])
    write_csv(target/'per_run.csv',all_rows)
    if scale_rows: write_csv(target/'per_scale.csv',scale_rows)
    # Only checkpoints produced by THIS shard; never shared source backbones.
    for path in model_files:
        if not path.resolve().is_relative_to((target/'temporary_training').resolve()):
            raise ValueError('unsafe checkpoint cleanup target')
        path.unlink()
    write_json(target/'complete.json',{'status':'COMPLETE','config_sha256':sha(config_path),
               'task':task,'dataset':dataset,'seed':seed,'rows':len(all_rows),'predictions':prediction_files,
               'per_run_sha256':sha(target/'per_run.csv'),'temporary_checkpoints_remaining':0,
               'node':socket.gethostname(),'device':torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})


def summarize(spec,config_path):
    root=Path(spec['output_root']); rows=[]
    for task in spec['tasks']:
        for dataset in spec['datasets']:
            for seed in spec['seeds']:
                directory=root/'shards'/task/dataset/f'seed{seed}'
                done=json.loads((directory/'complete.json').read_text())
                if done['config_sha256']!=sha(config_path) or sha(directory/'per_run.csv')!=done['per_run_sha256']:
                    raise ValueError('shard changed after completion')
                with (directory/'per_run.csv').open(newline='') as f: rows.extend(list(csv.DictReader(f)))
    write_csv(root/'per_run.csv',rows)
    fields=['f1','average_precision','iou','f1_drop_from_clean','average_precision_drop_from_clean','iou_drop_from_clean']
    table=[]
    for task in spec['tasks']:
        for condition in spec['corruptions']:
            for arm in sorted({r['arm'] for r in rows if r['task']==task}):
                subset=[r for r in rows if r['task']==task and r['condition']==condition['id'] and r['arm']==arm]
                row={'task':task,'condition':condition['id'],'arm':arm,'datasets':len(spec['datasets'])}
                for metric in fields:
                    row[metric]=float(np.mean([np.mean([float(r[metric]) for r in subset if r['dataset']==d]) for d in spec['datasets']]))
                table.append(row)
    write_csv(root/'robustness_table.csv',table)
    write_csv(root/'clean_comparison.csv',[r for r in table if r['condition']=='clean'])
    write_json(root/'summary.json',{'status':'COMPLETE_PENDING_INDEPENDENT_AUDIT','rows':len(rows),
               'config_sha256':sha(config_path),'noise_augmented_training':False,
               'capacity_matched_new_experiment_not_historical_main_table':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config/Verify_Task135_GeometricControls_1.1.yaml')
    p.add_argument('--mode',required=True,choices=['preflight','run','summarize'])
    p.add_argument('--job-index',type=int)
    args=p.parse_args(); spec=yaml.safe_load(Path(args.config).read_text())
    if args.mode=='preflight': preflight(spec,args.config)
    elif args.mode=='run': run_shard(spec,args.config,args.job_index)
    else: summarize(spec,args.config)


if __name__=='__main__': main()
