"""Paired frozen-benchmark replay of the objective neighbour encoder."""
from __future__ import annotations
import argparse
import gc
import hashlib
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
from FMT_Utils.FMTAllV2_3D import fmt_all_v2
from FMT_Utils.Task12Data_3D import feature_matrix
from FMT_Utils.Task12Evaluation_3D import (
    fit_kmeans_transform, calibrate_vortex_cluster, binary_cluster_metrics,
)
from FMT_Utils.RawPathline_3D import raw_pathline_representation, normalize_raw_train_eval
from FMT_Utils.GeometricControls_3D import pad_auxiliary
from experiments.Run_Task135_GeometricControls import (
    sources, load_records, sha, write_json, write_csv, load_residual, metrics,
)
from experiments.Verify_HighReVAE import _train
from experiments.Verify_Task3_FMTClassifier import _normalize_train_only, _loader, _predict
from experiments.Verify_Task3_FMTResidual import _train_one, _load_raw_model, _predict_components, _probabilities


def records(spec, task, dataset, role):
    phase = 'confirmation' if role == 'confirmation' and task != 'Task2' else 'development'
    # Task2 uses the fixed pathline caches and their frozen p95 reference labels.
    return load_records(spec, 'Task1' if task == 'Task2' else task, dataset,
                        phase, spec[task.lower()][role])


def evidence(rows):
    return [{'path': r['path'], 'sha256': sha(r['path']), 'ordinal': r['ordinal'],
             'identity': r['identity'], 'metadata': r['metadata'], **r['certificate']} for r in rows]


def features(rows, name, device):
    values=[]
    for r in rows:
        if name == 'fmt_all_v2':
            value=fmt_all_v2(r['raw'])
        else:
            adapter={'raw': r['raw'].reshape(len(r['raw']), -1),
                     'fmt': r['cached_fmt'], 'features': {}}
            value=feature_matrix(adapter, name, device)
        values.append(value)
    return np.concatenate(values)


def raw_array(rows):
    return np.concatenate([r['raw'] for r in rows])


def labels(rows):
    return np.concatenate([r['labels'] for r in rows]).astype(np.float32)


def objectivity_check(rows):
    x=rows[0]['raw'][:32].astype(np.float64)
    rng=np.random.default_rng(7097)
    q=[]
    for _ in range(x.shape[2]):
        rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
        rotation[:,0]*=np.linalg.det(rotation)
        q.append(rotation)
    y=np.einsum('tij,nktj->nkti', np.asarray(q), x)
    y+=rng.normal(size=(1,1,x.shape[2],3))*max(float(np.ptp(x)), 1.0)
    before,after=fmt_all_v2(x),fmt_all_v2(y)
    error=float(np.max(np.abs(before-after)))
    relative=float(np.linalg.norm(before-after)/max(np.linalg.norm(before), 1e-12))
    np.testing.assert_allclose(before, after, rtol=2e-5, atol=2e-5)
    return {'status': 'PASS', 'material_primitives': len(x),
            'max_absolute_error': error, 'relative_l2_error': relative,
            'transform': 'independent_SO3_and_translation_at_each_sample_float64'}


def prepare_task2(train, validation, arm, old_feature, device, new_feature='fmt_all_v2'):
    if arm == 'raw':
        a=raw_pathline_representation(raw_array(train).reshape(len(labels(train)),-1), 'center_relative')
        b=raw_pathline_representation(raw_array(validation).reshape(len(labels(validation)),-1), 'center_relative')
        x,y=normalize_raw_train_eval(a,b,'center_relative',32,'pre_group_rms')
        # The same helper is reused with the unchanged training array for test.
        return x,y,('raw',a)
    name=old_feature if arm == 'old_fmt' else new_feature
    a,b=features(train,name,device),features(validation,name,device)
    scaler=StandardScaler().fit(a)
    return scaler.transform(a).astype(np.float32),scaler.transform(b).astype(np.float32),(name,scaler)


def encode_test(rows, transform, model, device):
    name,normalizer=transform
    if name == 'raw':
        a=raw_pathline_representation(raw_array(rows).reshape(len(labels(rows)),-1),'center_relative')
        _,x=normalize_raw_train_eval(normalizer,a,'center_relative',32,'pre_group_rms')
    else:
        x=normalizer.transform(features(rows,name,device)).astype(np.float32)
    with torch.no_grad():
        return np.concatenate([model.encode(torch.from_numpy(x[s:s+4096]).to(device))[0].cpu().numpy()
                               for s in range(0,len(x),4096)])


def run(spec, config_path, task, dataset, seed):
    part=spec[task.lower()]
    if dataset not in spec['datasets'] or seed not in part['seeds']:
        raise ValueError('unregistered dataset/seed')
    target=Path(spec['output_root'])/'shards'/task/dataset/f'seed{seed}'
    if target.exists():
        raise FileExistsError(f'refusing to overwrite or silently resume {target}')
    target.mkdir(parents=True)
    device=torch.device('cpu' if task == 'Task1' else 'cuda')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('allocated GPU required')
    started={'task': task, 'dataset': dataset, 'seed': seed,
             'config_sha256': sha(config_path), 'runner_sha256': sha(__file__),
             'encoder_sha256': sha(Path(__file__).resolve().parents[1]/'FMT_Utils/FMTAllV2_3D.py'),
             'job': os.environ.get('SLURM_JOB_ID'), 'array_task': os.environ.get('SLURM_ARRAY_TASK_ID'),
             'node': socket.gethostname(), 'start_time': time.time(),
             'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
             'torch': torch.__version__, 'numpy': np.__version__}
    write_json(target/'started.json',started)
    train,validation=records(spec,task,dataset,'train'),records(spec,task,dataset,'validation')
    audit=objectivity_check(train)
    write_json(target/'input_audit.json',{'training': evidence(train), 'validation': evidence(validation), 'objectivity': audit})
    train_y,val_y=labels(train),labels(validation)
    models={}; frozen={}; checkpoints=[]
    if task == 'Task1':
        for arm in part['arms']:
            name=part['old_feature'] if arm == 'old_fmt' else part.get('new_feature',arm)
            x,y=features(train,name,device),features(validation,name,device)
            model=fit_kmeans_transform(x,part['pca_dim'],seed,part['kmeans_n_init'])
            cluster=calibrate_vortex_cluster(val_y,model.predict(y))
            models[arm]=(model,cluster,name)
            frozen[arm]={'vortex_cluster': cluster, 'input_dimension': x.shape[1],
                         'calibration': binary_cluster_metrics(val_y,model.predict(y),cluster)}
    elif task == 'Task2':
        source=EasyConfig()
        source.update({'task2': {'batch_size': part['batch_size'], 'weight_decay': part['weight_decay'],
                                'learning_rate': part['architecture']['learning_rate'],
                                'target_optimizer_steps': part['architecture']['optimizer_steps']}})
        for arm in part['arms']:
            x,y,transform=prepare_task2(train,validation,arm,part['old_feature'],device,
                                       part.get('new_feature','fmt_all_v2'))
            train_mu,val_mu,losses,model=_train(x,y,part['architecture'],source,seed,device,return_model=True)
            clusterer=KMeans(n_clusters=2,random_state=part['kmeans_seed'],n_init=part['kmeans_n_init']).fit(train_mu)
            cluster=calibrate_vortex_cluster(val_y,clusterer.predict(val_mu))
            models[arm]=(model,clusterer,cluster,transform)
            frozen[arm]={'vortex_cluster': cluster, 'input_dimension': x.shape[1], 'losses': losses,
                         'calibration': binary_cluster_metrics(val_y,clusterer.predict(val_mu),cluster)}
            print(f'TRAINED {task} {dataset} {seed} {arm}',flush=True)
    else:
        _,_,backbone_dir,_=sources(spec,task,dataset,'development')
        backbone_path=backbone_dir/f'{dataset}_raw_seed{seed}.pt'
        backbone=torch.load(backbone_path,map_location='cpu',weights_only=False)
        width=part['auxiliary_input_width']
        for arm in part['arms']:
            name=part['old_feature'] if arm == 'old_fmt' else part.get('new_feature',arm)
            x=(raw_array(train),pad_auxiliary(features(train,name,device),width),train_y)
            y=(raw_array(validation),pad_auxiliary(features(validation,name,device),width),val_y)
            x,y,_,stats=_normalize_train_only(x,y,None,raw_stats=backbone['normalization'])
            directory=target/'temporary_training'/arm
            directory.mkdir(parents=True)
            run_spec={'experiment':spec['experiment'],'model':spec['model'],'fusion':spec['fusion'],
                      'training':spec['training'],'auxiliary_source':'fmt','raw_checkpoint_dir':str(backbone_dir),
                      'raw_wide_parameter_count':148225,'raw_backbone_seed':seed,
                      'evaluation':{'test_enabled':False},'split':part}
            result=_train_one(run_spec,dataset,seed,(x,y,None),stats,device,directory)
            model,state=load_residual(result['checkpoint'],width,device)
            models[arm]=(model,state,name)
            checkpoints.append(Path(result['checkpoint']))
            frozen[arm]={'threshold':float(state['threshold']),'alpha':float(state['alpha']),
                         'best_epoch':int(state['best_epoch']),'normalization':state['normalization'],
                         'parameter_count':int(state['total_parameter_count']),
                         'trainable_parameter_count':int(state['trainable_residual_parameter_count']),
                         'backbone_sha256':sha(backbone_path),'training_result':result}
        if len({frozen[a]['parameter_count'] for a in part['arms']})!=1:
            raise ValueError('Task5 paired network capacity differs')
        raw_model,_=_load_raw_model(backbone_path,width,device)
        frozen['raw_backbone']={'threshold':float(backbone['threshold']),'backbone_sha256':sha(backbone_path)}

    # Persist all model choices before opening confirmation caches.
    write_json(target/'frozen_models.json',frozen)
    del train,validation
    gc.collect()
    test=records(spec,task,dataset,'confirmation')
    write_json(target/'confirmation_input_audit.json',evidence(test))
    test_y=labels(test)
    result_rows=[]; scale_rows=[]; scores={}; thresholds={}; predictions={}
    for arm,item in models.items():
        if task in ('Task1','Task2'):
            if task == 'Task1':
                model,cluster,name=item
                predictions[arm]=model.predict(features(test,name,device))
            else:
                model,clusterer,cluster,transform=item
                predictions[arm]=clusterer.predict(encode_test(test,transform,model,device))
            outcome=binary_cluster_metrics(test_y,predictions[arm],cluster)
            predictions[arm]=(predictions[arm]==cluster).astype(np.uint8)
        else:
            model,state,name=item
            stats=state['normalization']
            x=((raw_array(test)-stats['raw_mean'])/stats['raw_std']).astype(np.float32)
            f=((pad_auxiliary(features(test,name,device),width)-stats['fmt_mean'])/stats['fmt_std']).astype(np.float32)
            loader=_loader((x,f,test_y),1024,False,seed,True)
            _,r,a=_predict_components(model,loader,device)
            scores[arm]=_probabilities(r,a,float(state['alpha']),state['config']['model'])
            thresholds[arm]=float(state['threshold'])
            outcome=metrics(test_y,scores[arm],thresholds[arm])
        result_rows.append({'task':task,'dataset':dataset,'seed':seed,'arm':arm,'sample_count':len(test_y),**outcome})
    if task == 'Task5':
        stats=backbone['normalization']
        x=((raw_array(test)-stats['raw_mean'])/stats['raw_std']).astype(np.float32)
        loader=_loader((x,np.zeros((len(x),width),np.float32),test_y),1024,False,seed,True)
        _,scores['raw_backbone']=_predict(raw_model,loader,device)
        thresholds['raw_backbone']=float(backbone['threshold'])
        result_rows.append({'task':task,'dataset':dataset,'seed':seed,'arm':'raw_backbone',
                            **metrics(test_y,scores['raw_backbone'],thresholds['raw_backbone'])})
        scale_id=np.concatenate([r['scale_id'] for r in test])
        for arm in scores:
            predictions[arm]=(scores[arm]>=thresholds[arm]).astype(np.uint8)
            for scale in np.unique(scale_id):
                mask=scale_id==scale
                scale_rows.append({'task':task,'dataset':dataset,'seed':seed,'arm':arm,'scale_id':int(scale),
                                   **metrics(test_y[mask],scores[arm][mask],thresholds[arm])})
        write_csv(target/'per_scale.csv',scale_rows)
    np.savez_compressed(target/'predictions.npz',labels=test_y,
                        **{f'prediction_{a}':v for a,v in predictions.items()},
                        **{f'score_{a}':v for a,v in scores.items()})
    write_csv(target/'per_run.csv',result_rows)
    for checkpoint in checkpoints:
        if not checkpoint.resolve().is_relative_to((target/'temporary_training').resolve()):
            raise ValueError('unsafe cleanup path')
        checkpoint.unlink()
    if list((target/'temporary_training').rglob('*.pt')):
        raise RuntimeError('temporary checkpoint remains')
    write_json(target/'complete.json',{**started,'status':'COMPLETE','end_time':time.time(),
               'rows':len(result_rows),'per_run_sha256':sha(target/'per_run.csv'),
               'prediction_sha256':sha(target/'predictions.npz'),'temporary_checkpoints_remaining':0})
    print(json.dumps(result_rows),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--task',choices=['Task1','Task2','Task5'],required=True)
    parser.add_argument('--index',type=int,required=True)
    args=parser.parse_args()
    spec=json.loads(Path(args.config).read_text())
    seeds=spec[args.task.lower()]['seeds']
    dataset_index,seed_index=divmod(args.index,len(seeds))
    run(spec,args.config,args.task,spec['datasets'][dataset_index],seeds[seed_index])


if __name__=='__main__':
    main()
