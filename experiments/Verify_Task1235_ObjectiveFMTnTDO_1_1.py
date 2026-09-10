"""Paired performance experiment for the complete nTDO 1.1 descriptor.

Frozen training helpers are reused; the encoder and historical results are
never modified. Each shard freezes every arm before opening evaluation data.
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from DeepUtils.utils import EasyConfig
from FMT_Utils.objective_fmt_nTDO import objective_fmt_nTDO
from FMT_Utils.Task12Data_3D import feature_matrix
from FMT_Utils.GeometricControls_3D import validate_cache, pad_auxiliary
from FMT_Utils.Task12Evaluation_3D import fit_kmeans_transform, calibrate_vortex_cluster, binary_cluster_metrics
from FMT_Utils.RawPathline_3D import raw_pathline_representation, normalize_raw_train_eval
from experiments.Run_Task135_GeometricControls import load_records, sources, sha, write_json, write_csv, metrics, load_residual
from experiments.Verify_HighReVAE import _train as train_vae
from experiments.Verify_Task3_FMTClassifier import _train_one as train_raw, _normalize_train_only, _loader, _predict
from experiments.Verify_Task3_FMTResidual import (
    _train_one as train_residual, _load_raw_model, _predict_components,
    _probabilities, _apply_raw_pca_transform,
)
from FMT_Utils.PathlineClassifier_3D import PathlineBinaryClassifier3D


def ordinals(spec, task, dataset, role):
    return spec.get('dataset_splits', {}).get(dataset, {}).get(task, {}).get(role, spec[task.lower()][role])


def source_paths(spec, task, dataset, role):
    phase = 'confirmation' if role == 'confirmation' and task in ('Task1', 'Task5') else 'development'
    rebuilt = task == 'Task5' and dataset in spec['task5_rebuilt_datasets']
    if rebuilt:
        cache = Path(spec['output_root']) / spec['task5_cache_directory'] / phase / dataset
        expected = 6 if phase == 'development' else 4
    else:
        cache, _, _, expected = sources(spec, 'Task1' if task == 'Task2' else task, dataset, phase)
    paths = sorted(cache.glob('slice_*.npz'))
    if len(paths) != expected:
        raise ValueError(f'{cache}: expected {expected} slices, got {len(paths)}')
    return phase, rebuilt, [paths[i] for i in ordinals(spec, task, dataset, role)]


def records(spec, task, dataset, role):
    phase, rebuilt, paths = source_paths(spec, task, dataset, role)
    ids = ordinals(spec, task, dataset, role)
    if rebuilt:
        result = []
        for ordinal, path in zip(ids, paths):
            with np.load(path, allow_pickle=False) as z:
                r = validate_cache(z)
            assert float(r['metadata']['ivd_percentile']) == 95
            r.update(path=str(path), context=str(path), ordinal=ordinal)
            result.append(r)
    else:
        result = load_records(spec, 'Task1' if task == 'Task2' else task, dataset, phase, ids)
    if dataset in spec['cylinder_datasets']:
        assert all(float(r['metadata']['source_time']) >= 7.5 - 1e-6 for r in result)
    y=labels(result)
    assert 0 < y.sum() < len(y), f'{task}/{dataset}/{role}: both binary classes are required'
    return result


def evidence(rows):
    return [{'path': r['path'], 'sha256': sha(r['path']), 'ordinal': r['ordinal'],
             'identity': r['identity'], 'metadata': r['metadata'], **r['certificate']} for r in rows]


def raw_array(rows):
    return np.concatenate([r['raw'] for r in rows])


def labels(rows):
    return np.concatenate([r['labels'] for r in rows]).astype(np.float32)


def raw_vae_array(rows, arm):
    x = raw_array(rows)
    return raw_pathline_representation(x.reshape(len(x), -1), 'center_relative')


def features(rows, name, task, device):
    parts = []
    for r in rows:
        if name == 'objective_fmt_nTDO':
            value = objective_fmt_nTDO(r['raw'], num_freq=6)
            assert value.shape == (len(r['raw']), 369)
        else:
            adapter = {'raw': r['raw'].reshape(len(r['raw']), -1), 'fmt': r['cached_fmt'], 'features': {}}
            value = feature_matrix(adapter, name, device)
        assert np.isfinite(value).all()
        parts.append(value)
    return np.concatenate(parts).astype(np.float32)


def feature_name(spec, task, arm):
    return spec[task.lower()].get('features', spec['features'])[arm]


def preflight(spec, config_path):
    """Inspect metadata, not evaluation features or evaluation label values."""
    certificates = []
    for task in spec['tasks']:
        for dataset in spec['datasets']:
            sets = []
            for role in ('train', 'validation', 'confirmation'):
                _, _, paths = source_paths(spec, task, dataset, role)
                sets.append(set(map(str, paths)))
                for path in paths:
                    with np.load(path, allow_pickle=False) as z:
                        m = json.loads(str(z['metadata_json']))
                    if dataset in spec['cylinder_datasets']:
                        assert float(m['source_time']) >= 7.5 - 1e-6
                    certificates.append({'task': task, 'dataset': dataset, 'role': role,
                                         'path': str(path), 'sha256': sha(path), 'metadata': m})
            assert not any(sets[a] & sets[b] for a, b in ((0,1), (0,2), (1,2)))
            train = records(spec, task, dataset, 'train')
            for arm in spec[task.lower()]['arms']:
                if arm in ('raw_wide', 'raw_pca', 'fixed_task3_raw'):
                    continue
                x = features(train[:1], feature_name(spec, task, arm), task, 'cpu')
                assert len(x) == len(train[0]['raw'])
            assert len(labels(train)) > 369
            del train
            print(f'PREFLIGHT {task}/{dataset} PASS', flush=True)
    write_json(Path(spec['output_root'])/'preflight.json', {
        'status': 'PASS', 'config_sha256': sha(config_path), 'certificates': certificates,
        'evaluation_features_or_labels_read': False, 'evaluation_metrics_computed': False,
        'benchmark_limit': spec['protocol']['benchmark_limit'], 'end_time': time.time()})


def provenance(spec, config_path, task, dataset, seed, device):
    return {'experiment':spec['experiment'], 'task':task, 'dataset':dataset, 'seed':seed,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'config_sha256':sha(config_path), 'runner_sha256':sha(__file__),
        'encoder_sha256':sha('FMT_Utils/objective_fmt_nTDO.py'),
        'source_manifest_sha256':sha('SOURCE_MANIFEST.sha256'), 'job':os.environ.get('SLURM_JOB_ID'),
        'array_task':os.environ.get('SLURM_ARRAY_TASK_ID'), 'node':socket.gethostname(), 'start_time':time.time(),
        'device':torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',
        'torch':torch.__version__, 'numpy':np.__version__}


def run(spec, config_path, task, dataset, seed):
    part = spec[task.lower()]
    assert dataset in spec['datasets'] and seed in part['seeds']
    root = Path(spec['output_root'])
    pre = json.loads((root/'preflight.json').read_text())
    assert pre['status'] == 'PASS' and pre['config_sha256'] == sha(config_path)
    target = root/'shards'/task/dataset/f'seed{seed}'
    target.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    device = torch.device('cpu' if task == 'Task1' else spec.get('device', 'cuda'))
    if device.type == 'cuda': assert torch.cuda.is_available()
    started = provenance(spec,config_path,task,dataset,seed,device)
    write_json(target/'started.json',started)
    train, val = records(spec,task,dataset,'train'), records(spec,task,dataset,'validation')
    write_json(target/'input_audit.json',{'training':evidence(train),'validation':evidence(val)})
    ty, vy = labels(train), labels(val)
    models, frozen = {}, {}
    if task in ('Task1','Task2'):
        for arm in part['arms']:
            name = feature_name(spec,task,arm)
            if task == 'Task2' and arm == 'raw':
                a = raw_vae_array(train,arm)
                b = raw_vae_array(val,arm)
                x,y = normalize_raw_train_eval(a,b,'center_relative',32,'pre_group_rms')
                normalizer = ('raw',a)
            else:
                a,b = features(train,name,task,device), features(val,name,task,device)
                if task == 'Task2':
                    scaler = StandardScaler().fit(a)
                    x,y = scaler.transform(a).astype(np.float32),scaler.transform(b).astype(np.float32)
                    normalizer = ('feature',scaler)
                else: x,y = a,b
            if task == 'Task1':
                dim = None if x.shape[1] == 1 else part['pca_dim']
                model = fit_kmeans_transform(x,dim,seed,part['kmeans_n_init'])
                cluster = calibrate_vortex_cluster(vy,model.predict(y))
                models[arm] = (model,cluster,name)
                frozen[arm] = {'feature':name,'pca_dimension':dim,'input_dimension':x.shape[1],
                              'vortex_cluster':cluster,'calibration':binary_cluster_metrics(vy,model.predict(y),cluster)}
            else:
                x,y = pad_auxiliary(x,spec['input_width']),pad_auxiliary(y,spec['input_width'])
                settings = part['architecture']
                source = EasyConfig();source.update({'task2':{'batch_size':part['batch_size'],
                    'weight_decay':part['weight_decay'],'learning_rate':settings['learning_rate'],
                    'target_optimizer_steps':settings['optimizer_steps']}})
                mu,vm,losses,model = train_vae(x,y,settings,source,seed,device,return_model=True)
                km = KMeans(n_clusters=2,random_state=part['kmeans_seed'],n_init=part['kmeans_n_init']).fit(mu)
                cluster = calibrate_vortex_cluster(vy,km.predict(vm))
                models[arm] = (model,km,cluster,name,normalizer)
                frozen[arm] = {'feature':name,'input_dimension':x.shape[1],'vortex_cluster':cluster,
                              'losses':losses,'calibration':binary_cluster_metrics(vy,km.predict(vm),cluster)}
            print(f'TRAINED {task}/{dataset}/{seed}/{arm}',flush=True)
    else:
        width = part['auxiliary_input_width']
        raw_train,raw_val = raw_array(train),raw_array(val)
        tr = (raw_train,np.zeros((len(ty),width),np.float32),ty)
        va = (raw_val,np.zeros((len(vy),width),np.float32),vy)
        tr,va,_,stats = _normalize_train_only(tr,va,None)
        raw_dir = target/'temporary_training/raw_models'
        raw_spec = {'experiment':spec['experiment'],'model':spec['raw_model'],
                    'training':spec['training'],'evaluation':{'test_enabled':False}}
        for arm in ['raw','raw_wide']:
            outcome = train_raw(raw_spec,dataset,arm,seed,(tr,va,None),stats,device,raw_dir)
            state = torch.load(outcome['checkpoint'],map_location='cpu',weights_only=False)
            model = PathlineBinaryClassifier3D(variant=arm,fmt_dim=width,**spec['raw_model']).to(device)
            model.load_state_dict(state['state_dict']);model.eval()
            models[arm] = ('raw',model,state)
            frozen[arm] = {'threshold':float(state['threshold']),'best_epoch':state['best_epoch'],
                          'parameter_count':state['parameter_count'],'normalization':stats,'training_result':outcome}
        raw_checkpoint = raw_dir/'checkpoints'/f'{dataset}_raw_seed{seed}.pt'
        for arm in part['arms']:
            if arm in ('raw','raw_wide','fixed_task3_raw'): continue
            name = None if arm=='raw_pca' else feature_name(spec,task,arm)
            auxiliary_preprocess=None
            if arm == 'raw_pca':
                x,y,fs = tr,va,stats
            else:
                a=features(train,name,task,device)
                b=features(val,name,task,device)
                a=pad_auxiliary(a,width);b=pad_auxiliary(b,width)
                x,y,_,fs=_normalize_train_only((raw_train,a,ty),(raw_val,b,vy),None,raw_stats=stats)
            run_spec={'experiment':spec['experiment'],'model':spec['model'],'fusion':part['fusion'],
                      'training':spec['training'],'raw_checkpoint_dir':str(raw_dir/'checkpoints'),
                      'raw_backbone_seed':seed,'raw_wide_parameter_count':148225,
                      'auxiliary_source':'raw_pca' if arm=='raw_pca' else 'fmt',
                      'raw_pca_components':width,'raw_pca_random_state':7068,
                      'evaluation':{'test_enabled':False},'split':part}
            directory=target/'temporary_training'/arm;directory.mkdir(parents=True)
            outcome=train_residual(run_spec,dataset,seed,(x,y,None),fs,device,directory)
            model,state=load_residual(outcome['checkpoint'],width,device)
            models[arm]=('residual',model,state,name,auxiliary_preprocess)
            frozen[arm]={'feature':name,'input_dimension':width,'threshold':float(state['threshold']),
                         'alpha':float(state['alpha']),'best_epoch':state['best_epoch'],
                         'parameter_count':state['total_parameter_count'],
                         'trainable_parameter_count':state['trainable_residual_parameter_count'],
                         'normalization':state['normalization'],'backbone_sha256':sha(raw_checkpoint),
                         'training_result':outcome}
        assert len({frozen[a]['parameter_count'] for a in part['arms'] if a not in ('raw','raw_wide','fixed_task3_raw')})==1
        if task=='Task5':
            source=root/'shards/Task3'/dataset/f'seed{seed}/temporary_training/raw_models/checkpoints'/f'{dataset}_raw_seed{seed}.pt'
            model,state=_load_raw_model(source,width,device)
            models['fixed_task3_raw']=('raw',model,state)
            frozen['fixed_task3_raw']={'threshold':float(state['threshold']),
                'parameter_count':state['parameter_count'],'normalization':state['normalization'],
                'source_checkpoint_sha256':sha(source),'source_task':'Task3'}
    write_json(target/'frozen_models.json',frozen)
    frozen_at=time.time()
    del train,val;gc.collect()
    test=records(spec,task,dataset,'confirmation')
    write_json(target/'confirmation_input_audit.json',evidence(test))
    y=labels(test);predictions={};scores={};thresholds={};result_rows=[];scale_rows=[]
    for arm,item in models.items():
        if task=='Task1':
            model,cluster,name=item
            ids=model.predict(features(test,name,task,device));predictions[arm]=(ids==cluster).astype(np.uint8)
            outcome=binary_cluster_metrics(y,ids,cluster)
        elif task=='Task2':
            model,km,cluster,name,(kind,norm)=item
            if kind=='raw':
                values=raw_vae_array(test,arm)
                _,x=normalize_raw_train_eval(norm,values,'center_relative',32,'pre_group_rms')
            else:x=norm.transform(features(test,name,task,device)).astype(np.float32)
            x=pad_auxiliary(x,spec['input_width'])
            with torch.no_grad():
                mu=np.concatenate([model.encode(torch.from_numpy(x[i:i+4096]).to(device))[0].cpu().numpy() for i in range(0,len(x),4096)])
            ids=km.predict(mu);predictions[arm]=(ids==cluster).astype(np.uint8)
            outcome=binary_cluster_metrics(y,ids,cluster)
        else:
            kind,model,state,*rest=item;stats=state['normalization']
            x=((raw_array(test)-stats['raw_mean'])/stats['raw_std']).astype(np.float32)
            if kind=='raw':
                aux=np.zeros((len(y),part['auxiliary_input_width']),np.float32)
                _,prob=_predict(model,_loader((x,aux,y),1024,False,seed,True),device)
            else:
                if arm=='raw_pca':aux=_apply_raw_pca_transform(x,state['auxiliary_transform'])
                else:
                    aux=features(test,rest[0],task,device)
                    if rest[1] is not None:
                        scaler,pca=rest[1];aux=pca.transform(scaler.transform(aux)).astype(np.float32)
                    aux=pad_auxiliary(aux,part['auxiliary_input_width'])
                    aux=((aux-stats['fmt_mean'])/stats['fmt_std']).astype(np.float32)
                _,r,a=_predict_components(model,_loader((x,aux,y),1024,False,seed,True),device)
                prob=_probabilities(r,a,float(state['alpha']),state['config']['model'])
            scores[arm]=prob;thresholds[arm]=float(state['threshold'])
            predictions[arm]=(prob>=thresholds[arm]).astype(np.uint8)
            outcome=metrics(y,prob,thresholds[arm])
        result_rows.append({'task':task,'dataset':dataset,'seed':seed,'arm':arm,'sample_count':len(y),**outcome})
    if task=='Task5':
        scale=np.concatenate([r['scale_id'] for r in test])
        for arm,prob in scores.items():
            for value in np.unique(scale):
                mask=scale==value
                scale_rows.append({'task':task,'dataset':dataset,'seed':seed,'arm':arm,'scale_id':int(value),
                                   **metrics(y[mask],prob[mask],thresholds[arm])})
        write_csv(target/'per_scale.csv',scale_rows)
    np.savez_compressed(target/'predictions.npz',labels=y,
        **{f'prediction_{k}':v for k,v in predictions.items()},**{f'score_{k}':v for k,v in scores.items()},
        **({'scale_id':scale} if task=='Task5' else {}))
    write_json(target/'thresholds.json',thresholds)
    write_csv(target/'per_run.csv',result_rows)
    write_json(target/'complete.json',{**started,'status':'COMPLETE','end_time':time.time(),
        'models_frozen_before_test_time':frozen_at,'rows':len(result_rows),
        'prediction_sha256':sha(target/'predictions.npz'),'per_run_sha256':sha(target/'per_run.csv'),
        'checkpoint_policy':'only_in_experiment_dependency_chain; final_audit_removes_all'})
    print(json.dumps(result_rows),flush=True)


def audit_shard(target, expected_arms):
    """Recompute reported results from retained predictions, without models."""
    from sklearn.metrics import f1_score, average_precision_score
    target = Path(target)
    complete = json.loads((target/'complete.json').read_text())
    assert complete['per_run_sha256'] == sha(target/'per_run.csv')
    assert complete['prediction_sha256'] == sha(target/'predictions.npz')
    rows = list(csv.DictReader((target/'per_run.csv').open()))
    assert {r['arm'] for r in rows} == set(expected_arms)
    assert len(rows) == len(expected_arms)
    thresholds = json.loads((target/'thresholds.json').read_text())
    with np.load(target/'predictions.npz', allow_pickle=False) as z:
        y = z['labels']
        for row in rows:
            arm = row['arm']
            prediction = z['prediction_'+arm]
            assert len(prediction) == int(row['sample_count']) == len(y)
            assert abs(f1_score(y,prediction,zero_division=0)-float(row['f1'])) < 1e-10
            if 'score_'+arm in z:
                score = z['score_'+arm]
                assert np.isfinite(score).all()
                np.testing.assert_array_equal(prediction, score >= thresholds[arm])
                assert abs(average_precision_score(y,score)-float(row['average_precision'])) < 1e-10
    frozen = json.loads((target/'frozen_models.json').read_text())
    if complete['task']=='Task2':
        assert {v['input_dimension'] for v in frozen.values()} == {700}
        assert len({v['losses']['parameter_count'] for v in frozen.values()}) == 1
        assert len({v['losses']['completed_optimizer_steps'] for v in frozen.values()}) == 1
    if complete['task'] in ('Task3','Task5'):
        assert {frozen[a]['parameter_count'] for a in ('raw_pca','old_fmt','ntdo')} == {138818}
    return rows


def cleanup_checkpoints(directory):
    directory = Path(directory).resolve()
    deleted = []
    for extension in ('*.pt','*.pth','*.ckpt'):
        for path in directory.rglob(extension):
            resolved = path.resolve()
            resolved.relative_to(directory)
            if 'temporary_training' not in resolved.parts:
                raise ValueError(f'unexpected checkpoint location: {resolved}')
            deleted.append({'path':str(resolved),'sha256':sha(resolved)})
            resolved.unlink()
    return deleted


def audit(spec):
    root = Path(spec['output_root'])
    rows, summary, paired = [], [], []
    for task in spec['tasks']:
        for dataset in spec['datasets']:
            for seed in spec[task.lower()]['seeds']:
                target = root/'shards'/task/dataset/f'seed{seed}'
                rows.extend(audit_shard(target,spec[task.lower()]['arms']))
    for task in spec['tasks']:
        part = [r for r in rows if r['task']==task]
        for arm in spec[task.lower()]['arms']:
            values = [r for r in part if r['arm']==arm]
            item = {'task':task,'arm':arm,'runs':len(values)}
            for metric in ('f1','average_precision','iou','ari','nmi'):
                if all(r.get(metric) not in (None,'') for r in values):
                    v=np.array([float(r[metric]) for r in values])
                    item[metric+'_mean']=float(v.mean());item[metric+'_std']=float(v.std(ddof=1))
            summary.append(item)
        for baseline in spec[task.lower()]['arms']:
            if baseline=='ntdo':continue
            for dataset in spec['datasets']:
                d=[r for r in part if r['dataset']==dataset]
                delta=[float(next(r['f1'] for r in d if r['arm']=='ntdo' and int(r['seed'])==seed))-
                       float(next(r['f1'] for r in d if r['arm']==baseline and int(r['seed'])==seed))
                       for seed in spec[task.lower()]['seeds']]
                paired.append({'task':task,'dataset':dataset,'baseline':baseline,'ntdo_minus_baseline_f1':float(np.mean(delta))})
    write_csv(root/'per_run_metrics.csv',rows)
    write_csv(root/'summary.csv',summary)
    write_csv(root/'paired_f1.csv',paired)
    deleted=cleanup_checkpoints(root/'shards')
    write_json(root/'final_audit.json',{'status':'PASS','rows':len(rows),'shards':120,
        'summary_sha256':sha(root/'summary.csv'),'deleted_checkpoints':deleted,'end_time':time.time()})
    print(json.dumps(summary,indent=2),flush=True)


def smoke(spec):
    """Exercise all training, freezing, transfer and scoring paths on synthetic data."""
    import copy
    import tempfile
    from unittest.mock import patch
    torch.set_num_threads(2)
    rng=np.random.default_rng(9091)
    with tempfile.TemporaryDirectory(prefix='ntdo_smoke_') as directory:
        s=copy.deepcopy(spec);s['output_root']=directory;s['device']='cpu';s['datasets']=['synthetic']
        s['training']['max_epochs']=1;s['training']['patience']=1
        s['task2']['architecture']['optimizer_steps']=3
        for task in s['tasks']:s[task.lower()]['seeds']=[0]
        config=Path(directory)/'config.json';write_json(config,s)
        write_json(Path(directory)/'preflight.json',{'status':'PASS','config_sha256':sha(config)})
        data={}
        for role,n in [('train',480),('validation',128),('confirmation',128)]:
            raw=rng.normal(size=(n,7,32,3)).astype(np.float32)
            data[role]=[{'raw':raw,'cached_fmt':rng.normal(size=(n,161)).astype(np.float32),
                         'labels':(np.arange(n)%5==0).astype(np.float32),'scale_id':np.arange(n)%9}]
        def fake_records(s,task,dataset,role):
            if role=='confirmation':
                assert (Path(directory)/'shards'/task/dataset/'seed0/frozen_models.json').is_file()
            return data[role]
        def fake_provenance(s,c,t,d,seed,device):
            return {'task':t,'dataset':d,'seed':seed,'start_time':time.time(),'smoke_only':True}
        with patch(__name__+'.records',side_effect=fake_records), patch(__name__+'.evidence',return_value=[]), \
                patch(__name__+'.provenance',side_effect=fake_provenance):
            for task in s['tasks']:
                run(s,str(config),task,'synthetic',0)
                audit_shard(Path(directory)/'shards'/task/'synthetic/seed0',s[task.lower()]['arms'])
        cleanup_checkpoints(Path(directory)/'shards')
    write_json(Path(spec['output_root'])/'smoke.json',{'status':'PASS','scientific_results':False,'end_time':time.time()})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config/Verify_Task1235_ObjectiveFMTnTDO_1.1.json')
    parser.add_argument('--phase',choices=['build','preflight','smoke','Task1','Task2','Task35','audit'],required=True)
    parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args();spec=json.loads(Path(args.config).read_text())
    if args.phase=='build':
        from experiments.Build_Task5_Multiscale_Cache import build
        import yaml
        import netCDF4 as nc
        from FMT_Utils.NetCDF_window_3D import _axis_dimension, _coordinate
        dataset=spec['task5_rebuilt_datasets'][args.index]
        cache_spec=yaml.safe_load(Path(spec['late_cache_config']).read_text())
        item=next(d for d in cache_spec['datasets'] if d['id']==dataset)
        source=next(Path(p) for p in item['paths'] if Path(p).is_file())
        with nc.Dataset(source) as data:
            tdim=_axis_dimension(data,'t')
            for phase in ('development','confirmation'):
                indices=cache_spec['phases'][phase]['time_indices_by_dataset'][dataset]
                actual=_coordinate(data,tdim,np.asarray(indices))
                np.testing.assert_allclose(actual,spec['task5_source_times'][phase],atol=1e-6)
                names=next(names for names in [('u','v','w'),('velocity_x','velocity_y','velocity_z'),
                      ('Component1','Component2','Component3')] if all(n in data.variables for n in names))
                for index in sorted({i for first in indices for i in range(first,first+14)}):
                    for name in names:
                        v=data.variables[name]
                        a=np.ma.asarray(v[tuple(index if d==tdim else slice(None,None,8) for d in v.dimensions)])
                        assert a.count()>0 and np.isfinite(a.compressed()).all(), (source,name,index)
        for phase in ('development','confirmation'):build(spec['late_cache_config'],phase,dataset)
    elif args.phase=='preflight':preflight(spec,args.config)
    elif args.phase=='smoke':smoke(spec)
    elif args.phase=='audit':audit(spec)
    else:
        tasks=['Task3','Task5'] if args.phase=='Task35' else [args.phase]
        seeds=spec[tasks[0].lower()]['seeds'];d,i=divmod(args.index,len(seeds))
        dataset,seed=spec['datasets'][d],seeds[i]
        for task in tasks:run(spec,args.config,task,dataset,seed)
        # Task3 checkpoints remain only until fixed-scale Task5 transfer completes.
        for task in tasks:
            target=Path(spec['output_root'])/'shards'/task/dataset/f'seed{seed}'
            audit_shard(target,spec[task.lower()]['arms'])
        deleted=[]
        for task in tasks:
            deleted.extend(cleanup_checkpoints(Path(spec['output_root'])/'shards'/task/dataset/f'seed{seed}'))
        write_json(Path(spec['output_root'])/'chain_audits'/args.phase/dataset/f'seed{seed}.json',
                   {'status':'PASS','deleted_checkpoints':deleted,'end_time':time.time()})


if __name__=='__main__':main()

