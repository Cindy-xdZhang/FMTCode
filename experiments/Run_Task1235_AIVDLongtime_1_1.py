"""Paired full-path versus first-three-sample IVD experiments on Ibex."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from DeepUtils.utils import EasyConfig
from FMT_Utils.Task12Data_3D import feature_matrix
from FMT_Utils.GeometricControls_3D import validate_cache, pad_auxiliary
from FMT_Utils.Task12Evaluation_3D import fit_kmeans_transform, calibrate_vortex_cluster, binary_cluster_metrics
from FMT_Utils.RawPathline_3D import raw_pathline_representation, normalize_raw_train_eval
from experiments.Run_Task135_GeometricControls import load_records, sha, write_json, write_csv, metrics, load_residual
from experiments.Verify_FMTAllV2_3D import evidence, labels, raw_array
from experiments.Verify_HighReVAE import _train as train_vae
from experiments.Verify_Task3_FMTClassifier import (
    _train_one as train_raw, _normalize_train_only, _loader, _predict,
)
from experiments.Verify_Task3_FMTResidual import (
    _train_one as train_residual, _load_raw_model, _predict_components,
    _probabilities, _apply_raw_pca_transform,
)
from FMT_Utils.PathlineClassifier_3D import PathlineBinaryClassifier3D


def ordinals(spec, task, dataset, role):
    return spec.get('dataset_splits', {}).get(dataset, {}).get(task, {}).get(role, spec[task.lower()][role])


def records(spec, task, dataset, role):
    phase = 'confirmation' if role == 'confirmation' and task in ('Task1', 'Task5') else 'development'
    selected = ordinals(spec, task, dataset, role)
    if task == 'Task5' and dataset in spec['cylinder_datasets']:
        directory = Path(spec['task5_late_cache'])/f'{phase}_cache'/dataset
        files = sorted(directory.glob('slice_*.npz'))
        assert len(files) == (6 if phase == 'development' else 2)
        result = []
        for ordinal in selected:
            with np.load(files[ordinal], allow_pickle=False) as archive:
                r = validate_cache(archive)
            assert r['metadata']['ivd_definition'] == 'whole_field_ivd_percentile'
            assert r['metadata']['ivd_percentile'] == 95
            r.update(path=str(files[ordinal]), context=str(files[ordinal]), ordinal=ordinal)
            result.append(r)
    else:
        result = load_records(spec, 'Task1' if task == 'Task2' else task, dataset, phase, selected)
    if dataset in spec['cylinder_datasets']:
        assert all(float(r['metadata']['source_time']) >= spec['minimum_cylinder_time'] for r in result)
    return result


def features(rows, name, task, device):
    result = []
    for r in rows:
        raw = r['raw']
        if name in ('aivd1w3_dft', 'aivd1w3_dft_longtime') and task == 'Task5':
            # Never subtract means across different physical times. Shared
            # time grids may pool different spatial offsets, with equal weights.
            _, groups = np.unique(r['times'], axis=0, return_inverse=True)
            value = np.empty((len(raw), 1), np.float32)
            for group in np.unique(groups):
                mask = groups == group
                assert np.sum(mask) > 1
                adapter = {'raw': raw[mask].reshape(np.sum(mask), -1), 'features': {}}
                value[mask] = feature_matrix(adapter, name, device)
        else:
            adapter = {'raw': raw.reshape(len(raw), -1), 'fmt': r['cached_fmt'], 'features': {}}
            value = feature_matrix(adapter, name, device)
        assert np.isfinite(value).all()
        result.append(value)
    return np.concatenate(result)


def feature_name(spec, task, arm):
    return spec['task5']['old_feature'] if task == 'Task5' and arm == 'old_fmt' else spec['features'][arm]


def preflight(spec, config_path, tasks):
    torch.set_num_threads(4)
    certificates, checks = [], []
    for task in tasks:
        for dataset in spec['datasets']:
            roles = {role: records(spec, task, dataset, role) for role in ('train', 'validation', 'confirmation')}
            sets = [{r['path'] for r in rows} for rows in roles.values()]
            assert not any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2)))
            for role, rows in roles.items():
                for r in rows:
                    if task == 'Task5':
                        expected_set = 'train' if role == 'train' else 'validation' if role == 'validation' else 'confirmation'
                        assert r['metadata']['scale_set'] == expected_set
                    certificates.append({'task':task, 'dataset':dataset, 'role':role,
                                         **evidence([r])[0]})
            if task == 'Task5' and dataset in spec['cylinder_datasets']:
                ranges = []
                for rows in roles.values():
                    ranges.append((min(r['metadata']['source_time'] for r in rows),
                        max(r['metadata']['source_time']+(r['metadata']['frame_count']-1)*r['metadata']['source_time_step'] for r in rows)))
                assert ranges[0][1] < ranges[1][0] and ranges[1][1] < ranges[2][0], ranges
            sample = roles['train'][:1]
            short = features(sample, spec['features']['short'], task, 'cpu')
            long = features(sample, spec['features']['long'], task, 'cpu')
            assert short.shape == long.shape == (len(sample[0]['raw']), 1)
            checks.append({'task':task,'dataset':dataset,'count':len(short),
                           'short_range':[float(short.min()),float(short.max())],
                           'long_range':[float(long.min()),float(long.max())],
                           'same_physical_time_groups':len(np.unique(sample[0]['times'],axis=0))})
            print(checks[-1], flush=True)
    suffix = 'task5' if tasks == ['Task5'] else 'fixed'
    write_json(Path(spec['output_root'])/f'preflight_{suffix}.json', {
        'status':'PASS','config_sha256':sha(config_path), 'source_manifest_sha256':sha('SOURCE_MANIFEST.sha256'),
        'tasks':tasks,'certificates':certificates,'feature_checks':checks,
        'test_metrics_computed':False,'test_features_computed':False,
        'node':socket.gethostname(),'job':os.environ.get('SLURM_JOB_ID'),'end_time':time.time()})


def run(spec, config_path, task, dataset, seed):
    part = spec[task.lower()]
    assert dataset in spec['datasets'] and seed in part['seeds']
    root = Path(spec['output_root'])
    pre = json.loads((root/f"preflight_{'task5' if task=='Task5' else 'fixed'}.json").read_text())
    assert pre['status'] == 'PASS' and pre['config_sha256'] == sha(config_path)
    target = root/'shards'/task/dataset/f'seed{seed}'
    target.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    device = torch.device('cpu' if task == 'Task1' else 'cuda')
    if device.type == 'cuda': assert torch.cuda.is_available()
    started = {'experiment':spec['experiment'],'task':task,'dataset':dataset,'seed':seed,
        'base_commit':spec['base_commit'],'config_sha256':sha(config_path),'runner_sha256':sha(__file__),
        'source_manifest_sha256':sha('SOURCE_MANIFEST.sha256'),'job':os.environ.get('SLURM_JOB_ID'),
        'array_task':os.environ.get('SLURM_ARRAY_TASK_ID'),'node':socket.gethostname(),'start_time':time.time(),
        'device':torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',
        'torch':torch.__version__,'numpy':np.__version__}
    write_json(target/'started.json',started)
    train, val = records(spec,task,dataset,'train'), records(spec,task,dataset,'validation')
    write_json(target/'input_audit.json',{'training':evidence(train),'validation':evidence(val)})
    ty, vy = labels(train), labels(val)
    models, frozen = {}, {}
    if task in ('Task1','Task2'):
        for arm in part['arms']:
            name = feature_name(spec,task,arm)
            if task == 'Task2' and arm == 'raw':
                a = raw_pathline_representation(raw_array(train).reshape(len(ty),-1),'center_relative')
                b = raw_pathline_representation(raw_array(val).reshape(len(vy),-1),'center_relative')
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
            if arm == 'raw_pca':
                x,y,fs = tr,va,stats
            else:
                a=pad_auxiliary(features(train,name,task,device),width)
                b=pad_auxiliary(features(val,name,task,device),width)
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
            models[arm]=('residual',model,state,name)
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
                values=raw_pathline_representation(raw_array(test).reshape(len(y),-1),'center_relative')
                _,x=normalize_raw_train_eval(norm,values,'center_relative',32,'pre_group_rms')
            else:x=norm.transform(features(test,name,task,device)).astype(np.float32)
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
                    aux=pad_auxiliary(features(test,rest[0],task,device),part['auxiliary_input_width'])
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config/Verify_AIVDLongtime_1.1.json')
    parser.add_argument('--task',choices=['Task1','Task2','Task3','Task5'])
    parser.add_argument('--index',type=int)
    parser.add_argument('--preflight',choices=['fixed','task5'])
    args=parser.parse_args();spec=json.loads(Path(args.config).read_text())
    if args.preflight:
        preflight(spec,args.config,['Task5'] if args.preflight=='task5' else ['Task1','Task2','Task3'])
    else:
        seeds=spec[args.task.lower()]['seeds'];d,s=divmod(args.index,len(seeds))
        run(spec,args.config,args.task,spec['datasets'][d],seeds[s])


if __name__=='__main__':main()
