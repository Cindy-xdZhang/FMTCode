"""Paired, preregistered mechanism diagnostic on an already used benchmark."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, silhouette_score
from DeepUtils.utils import EasyConfig
from FMT_Utils.ObjectivityMechanism_3D import ARMS, WIDTH, feature_set, pad, gram_series, observe, analytic_checks
from FMT_Utils.Task12Evaluation_3D import fit_kmeans_transform, calibrate_vortex_cluster, binary_cluster_metrics
from FMT_Utils.RawPathline_3D import normalize_raw_train_eval
from experiments.Run_Task135_GeometricControls import load_records, write_json, write_csv, sha
from experiments.Verify_HighReVAE import _train

DEFAULT='config/Verify_FMTObjectivityMechanism_1.3.json'


def records(spec,task,dataset,role):
    ordinals=spec.get('dataset_splits',{}).get(dataset,{}).get(task,{}).get(role,spec[task.lower()][role])
    phase='confirmation' if task=='Task1' and role=='confirmation' else 'development'
    rows=load_records(spec,'Task1',dataset,phase,ordinals)
    if dataset in spec['cylinder_datasets']:
        assert all(float(r['metadata']['source_time'])>=7.5 for r in rows)
    return rows


def inputs(rows):
    values={k:[] for k in ARMS}
    for r in rows:
        for key,x in feature_set(r['raw'],r['cached_fmt']).items():
            values[key].append(x)
    return {k:np.concatenate(v) for k,v in values.items()}


def evidence(rows):
    return [dict(path=r['path'],sha256=sha(r['path']),identity=r['identity'],
                 source_time=r['metadata']['source_time'],n=len(r['raw'])) for r in rows]


def encode(model,x,device):
    with torch.no_grad():
        return np.concatenate([model.encode(torch.from_numpy(x[i:i+2048]).to(device))[0].cpu().numpy()
                               for i in range(0,len(x),2048)])


def normalize(a,b,arm):
    if arm=='raw':
        x,y=normalize_raw_train_eval(a,b,'center_relative',32,'pre_group_rms')
        return pad(x),pad(y),('raw',a)
    scaler=StandardScaler().fit(a)
    return pad(scaler.transform(a)),pad(scaler.transform(b)),('scaled',scaler)


def apply_normalization(x,state):
    kind,obj=state
    if kind=='raw':
        _,y=normalize_raw_train_eval(obj,x,'center_relative',32,'pre_group_rms')
    else:
        y=obj.transform(x)
    return pad(y)


def score_silhouette(distance,clusters):
    n=len(np.unique(clusters))
    return float(silhouette_score(distance,clusters,metric='precomputed')) if 1<n<len(clusters) else None


def event(spec,phase,state,exit_code=None):
    item=dict(experiment=spec['experiment'],phase=phase,state=state,exit_code=exit_code,
              time=datetime.datetime.now().astimezone().isoformat(),node=socket.gethostname(),
              job=os.getenv('SLURM_ARRAY_JOB_ID',os.getenv('SLURM_JOB_ID','local')),
              index=os.getenv('SLURM_ARRAY_TASK_ID',''),
              config_sha256=sha(DEFAULT),source_manifest_sha256=sha('SOURCE_MANIFEST.sha256') if Path('SOURCE_MANIFEST.sha256').exists() else None,
              device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    root=Path(spec['output_root']);root.mkdir(parents=True,exist_ok=True)
    name=f"{item['job']}_{item['index']}_{phase}_{state}.json"
    write_json(root/'events'/name,item)
    with Path('docs/ibex_run_registry.md').open('a',encoding='utf-8') as f:
        if os.name!='nt':
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX)
        f.write('\n- '+spec['experiment']+' event '+json.dumps(item)+'\n')
        f.flush();os.fsync(f.fileno())
    print(json.dumps(item),flush=True)


def preflight(spec):
    root=Path(spec['output_root'])
    checks=analytic_checks();write_json(root/'analytic_checks.json',checks)
    certificates=[]
    for task in spec['tasks']:
        for dataset in spec['datasets']:
            groups=[records(spec,task,dataset,role) for role in ['train','validation','confirmation']]
            sets=[{r['path'] for r in g} for g in groups]
            assert not any(sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)])
            certificates.append(dict(task=task,dataset=dataset,
                                     roles={role:evidence(g) for role,g in zip(['train','validation','confirmation'],groups)}))
            # Full slice preserves the frozen kinematic mean context.
            r=groups[0][0];v=feature_set(r['raw'],r['cached_fmt'])
            for arm in ARMS:
                assert pad(v[arm]).shape==(len(r['raw']),WIDTH)
            # Cached FMT was computed on uncentred coordinates; raw storage was
            # subsequently centred in float32. Do not silently equate the two.
            from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
            old=pathline_dft_features_3d(torch.from_numpy(r['raw']),neighbor_scale=1.0,neighbor_weight=1.0)
            np.testing.assert_array_equal(old,v['old_recomputed_full'][:,:161])
            np.testing.assert_array_equal(r['cached_fmt'],v['old_full'][:,:161])
            np.testing.assert_array_equal(v['old_full'][:,23:],v['old_no_center'][:,23:])
            difference=np.abs(old-r['cached_fmt'])
            certificates[-1]['cache_recompute_difference']={
                'max_absolute':float(difference.max()),
                'fraction_outside_previous_tolerance':float((difference>(2e-4+2e-4*np.abs(r['cached_fmt']))).mean()),
                'relative_l2':float(np.linalg.norm(old-r['cached_fmt'])/max(np.linalg.norm(r['cached_fmt']),1e-12)),
                'interpretation':'Distinct registered arms; discrepancy is measured, not passed as numerical equality.'}
            certificates[-1]['widths']={k:x.shape[1] for k,x in v.items()}
            if task=='Task1':
                # A separate, fixed 256-primitive diagnostic context, recomputed
                # consistently before and after; no claim to whole-field IVD.
                sample=r['raw'][:256].astype(np.float64)
                before=feature_set(sample)
                drift=[]
                for name,changed in [('global_rotation',observe(sample)),
                    ('translation',observe(sample,translation=True)),
                    ('independent_local_rotation_NOT_observer',observe(sample,local=True))]:
                    after=feature_set(changed)
                    for arm in ARMS:
                        delta=float(np.linalg.norm(after[arm]-before[arm]))
                        drift.append(dict(transform=name,arm=arm,absolute_l2=delta,
                                          relative_l2=delta/max(float(np.linalg.norm(before[arm])),1e-12)))
                    np.testing.assert_allclose(after['gram6'],before['gram6'],rtol=2e-5,atol=2e-5)
                certificates[-1]['drift_256_primitives']=drift
            print('PREFLIGHT',task,dataset,flush=True)
    write_json(root/'preflight.json',dict(status='PASS',certificates=certificates,
                                         confirmation_performance_computed=False))


def run(spec,task,index,smoke=False):
    part=spec[task.lower()];seeds=part['seeds']
    dataset=spec['datasets'][index//len(seeds)];seed=seeds[index%len(seeds)]
    root=Path(spec['output_root'])/('smoke' if smoke else 'shards')/task/dataset/f'seed{seed}'
    root.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda' if task=='Task2' and not smoke else 'cpu')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU required')
    torch.set_num_threads(4)
    train=records(spec,task,dataset,'train');val=records(spec,task,dataset,'validation')
    if smoke:
        # Smoke is executable validation only; no benchmark inference.
        train=train[:1];val=val[:1]
        train=[dict(r,raw=r['raw'][:96],cached_fmt=r['cached_fmt'][:96],labels=r['labels'][:96]) for r in train]
        val=[dict(r,raw=r['raw'][:96],cached_fmt=r['cached_fmt'][:96],labels=r['labels'][:96]) for r in val]
    a,b=inputs(train),inputs(val)
    ty=np.concatenate([r['labels'] for r in train]);vy=np.concatenate([r['labels'] for r in val])
    write_json(root/'inputs.json',dict(train=evidence(train),validation=evidence(val)))
    models={};frozen={}
    arch=dict(part.get('architecture',{}))
    if smoke and task=='Task2':arch['optimizer_steps']=3
    source=EasyConfig();source.update({'task2':dict(batch_size=part.get('batch_size',256),
        weight_decay=part.get('weight_decay',1e-5),learning_rate=3e-4,target_optimizer_steps=7000)})
    for arm in ARMS:
        if task=='Task1':
            for dimension in [8,None]:
                key=arm+('_pca8' if dimension else '_no_pca')
                # All arms use the same feature width; train-only transformations.
                m=fit_kmeans_transform(pad(a[arm]),dimension,seed,20)
                cluster=calibrate_vortex_cluster(vy,m.predict(pad(b[arm])))
                models[key]=(arm,m,cluster)
                frozen[key]=dict(cluster=cluster,native_width=a[arm].shape[1],input_width=WIDTH,
                                  retained_variance=float(m.pca.explained_variance_ratio_.sum()) if m.pca else 1.)
        else:
            x,y,state=normalize(a[arm],b[arm],arm)
            mu,vmu,loss,m=_train(x,y,arch,source,seed,device,return_model=True)
            k=KMeans(n_clusters=2,n_init=20,random_state=7068).fit(mu)
            cluster=calibrate_vortex_cluster(vy,k.predict(vmu))
            models[arm]=(arm,m,k,cluster,state)
            frozen[arm]=dict(cluster=cluster,native_width=a[arm].shape[1],input_width=WIDTH,loss=loss)
        print('FROZEN',task,dataset,seed,arm,flush=True)
    write_json(root/'frozen.json',frozen)
    if smoke:
        write_json(root/'complete.json',dict(status='PASS',models=len(models),scientific_result=False))
        return
    # No test features/performance are read before all arms are frozen.
    test=records(spec,task,dataset,'confirmation');c=inputs(test)
    y=np.concatenate([r['labels'] for r in test]).astype(np.uint8)
    write_json(root/'test_inputs.json',evidence(test))
    ids=np.sort(np.random.default_rng(19091).choice(len(y),min(768,len(y)),replace=False))
    from sklearn.metrics import pairwise_distances
    common=np.concatenate([gram_series(r['raw']).reshape(len(r['raw']),-1) for r in test])[ids]
    distance=pairwise_distances(common);np.fill_diagonal(distance,0)
    predictions={};result=[]
    boundaries=np.cumsum([0]+[len(r['raw']) for r in test])
    for key,item in models.items():
        arm=item[0]
        if task=='Task1':
            _,m,cluster=item
            z=m.transform(pad(c[arm]));clusters=m.model.predict(z)
        else:
            _,m,k,cluster,state=item
            z=encode(m,apply_normalization(c[arm],state),device);clusters=k.predict(z)
        pred=(clusters==cluster).astype(np.uint8);predictions[key]=pred
        d=pairwise_distances(z[ids]);np.fill_diagonal(d,0)
        metrics=binary_cluster_metrics(y,clusters,cluster)
        null=[];rng=np.random.default_rng(29091)
        for repeat in range(100):
            yp=y.copy()
            for l,r in zip(boundaries[:-1],boundaries[1:]):yp[l:r]=rng.permutation(y[l:r])
            null.append(float(f1_score(yp,pred,zero_division=0)))
        result.append(dict(task=task,dataset=dataset,seed=seed,arm=arm,model=key,n=len(y),
                           silhouette_own=score_silhouette(d,clusters[ids]),
                           silhouette_common_gram=score_silhouette(distance,clusters[ids]),
                           shuffled_label_f1_mean=float(np.mean(null)),
                           shuffled_label_f1_p95=float(np.quantile(null,.95)),**metrics))
    np.savez_compressed(root/'predictions.npz',labels=y,analysis_ids=ids,**predictions)
    write_csv(root/'metrics.csv',result)
    write_json(root/'complete.json',dict(status='COMPLETE',rows=len(result),
        device=torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',
        completed=datetime.datetime.now().astimezone().isoformat(),checkpoint_count=0))


def audit(spec):
    import csv
    allrows=[]
    for task in spec['tasks']:
        for dataset in spec['datasets']:
            for seed in spec[task.lower()]['seeds']:
                root=Path(spec['output_root'])/'shards'/task/dataset/f'seed{seed}'
                assert json.loads((root/'complete.json').read_text())['status']=='COMPLETE'
                frozen=json.loads((root/'frozen.json').read_text())
                rows=list(csv.DictReader((root/'metrics.csv').open()))
                assert len(rows)==len(ARMS)*(2 if task=='Task1' else 1)
                with np.load(root/'predictions.npz') as p:
                    for row in rows:
                        for metric,value in binary_cluster_metrics(p['labels'],p[row['model']],1).items():
                            np.testing.assert_allclose(float(row[metric]),value,atol=1e-12)
                if task=='Task2':
                    assert len({v['loss']['parameter_count'] for v in frozen.values()})==1
                    assert all(v['loss']['completed_optimizer_steps']==7000 for v in frozen.values())
                assert not list(root.rglob('*.pt'))
                allrows.extend(rows)
    out=Path(spec['output_root']);write_csv(out/'per_run_metrics.csv',allrows)
    summaries=[]
    for task in spec['tasks']:
        keys=sorted({r['model'] for r in allrows if r['task']==task})
        for key in keys:
            rows=[r for r in allrows if r['task']==task and r['model']==key]
            summaries.append(dict(task=task,model=key,rows=len(rows),
                                  macro_f1=float(np.mean([float(r['f1']) for r in rows]))))
    write_csv(out/'summary.csv',summaries)
    write_json(out/'independent_audit.json',dict(status='PASS',rows=len(allrows),
        expected=10*3*len(ARMS)*3,scope='Prediction-based classification metrics, coverage, capacity and steps; silhouette retained as diagnostics.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=DEFAULT)
    p.add_argument('--phase',choices=['preflight','Task1','Task2','smoke','audit'],required=True)
    p.add_argument('--index',type=int,default=0)
    args=p.parse_args();spec=json.loads(Path(args.config).read_text())
    torch.set_num_threads(4)
    event(spec,args.phase,'RUNNING')
    try:
        if args.phase=='preflight':preflight(spec)
        elif args.phase=='audit':audit(spec)
        elif args.phase=='smoke':
            run(spec,'Task1',0,True);run(spec,'Task2',0,True)
        else:run(spec,args.phase,args.index)
    except BaseException:
        event(spec,args.phase,'FAILED',1);raise
    event(spec,args.phase,'COMPLETED',0)


if __name__=='__main__':main()
