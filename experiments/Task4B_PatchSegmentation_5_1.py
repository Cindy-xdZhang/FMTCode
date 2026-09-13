"""Version 5.1: disjoint local volumes, pooled FMT pointwise segmentation."""
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
from FMT_Utils.Task4B_PatchSplit_3D import make_manifest, audit_manifest, local_step, overlaps


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4194304),b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def identity():
    return dict(host=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),
                script_sha256=sha(__file__),split_code_sha256=sha('FMT_Utils/Task4B_PatchSplit_3D.py'))


def plan(spec,config):
    out=Path(spec['output']); out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        raise RuntimeError('Frozen manifest exists')
    original=json.loads((Path(spec['labels_source'])/'summary.json').read_text())
    report=dict(version=spec['version'],config_sha256=sha(config),flows={},**identity())
    for code,name in enumerate(('channel','tbl')):
        source=Path(spec['labels_source'])/f'{name}_groundtruth.npz'
        assert sha(source)==original['flows'][name]['groundtruth_sha256']
        with np.load(source) as d:
            assert float(d['threshold'])==spec['thresholds'][name]
            opts={k:v for k,v in spec['split'].items() if k!='policy'}
            opts['seed']+=code
            patches,audit=make_manifest(d['vortex_ids'],**opts)
            report['flows'][name]=dict(patches=patches,audit=audit,label_sha256=sha(source),
                low_xyz=d['domain_min_xyz'].tolist(),high_xyz=d['domain_max_xyz'].tolist(),
                resolution_xyz=d['resolution_xyz'].tolist())
        print(name,audit,flush=True)
    (out/'config_snapshot.json').write_bytes(Path(config).read_bytes())
    write(out/'manifest.json',report)


def patch_candidates(labels,ids,patch,rng,cap):
    low=np.array(patch['low']); high=np.array(patch['high'])
    slices=tuple(slice(a,b) for a,b in zip(low,high))
    index=np.argwhere(labels[slices]>=0)+low
    flat=np.ravel_multi_index(index.T,labels.shape) if len(index) else np.array([],dtype=np.int64)
    if patch['split']=='train':
        hairpin=flat[ids.ravel()[flat]>0]; ordinary=flat[ids.ravel()[flat]==0]
        if len(ordinary)>cap:
            ordinary=rng.choice(ordinary,cap,replace=False)
        flat=np.sort(np.r_[hairpin,ordinary])
    return flat


def integrate_local(field,seeds,h,inner_low,inner_high,steps,offset_ratio,minimum_speed):
    """Seven RK4 trajectories; every native interpolation node stays in patch."""
    offsets=np.r_[np.zeros((1,3)),np.eye(3)*offset_ratio,-np.eye(3)*offset_ratio]
    initial=np.broadcast_to(offsets,(len(seeds),7,3)).copy()
    qlow=np.full((len(seeds),3),np.inf); qhigh=np.full((len(seeds),3),-np.inf)

    def direction(q):
        physical=seeds[:,None,:]+q*h[:,None,None]
        finite=np.isfinite(physical).all(-1)
        inside=((physical>=inner_low[:,None,:]-1e-11)&(physical<=inner_high[:,None,:]+1e-11)).all(-1)
        assert np.all(~finite|inside),'Primitive query escaped its native patch domain'
        velocity=np.full_like(physical,np.nan)
        velocity[finite]=field.velocity(physical[finite])
        speed=np.linalg.norm(velocity,axis=-1,keepdims=True)
        valid=np.isfinite(speed)&(speed>minimum_speed)
        result=np.full_like(velocity,np.nan)
        np.divide(velocity,speed,out=result,where=valid)
        for axis in range(3):
            values=physical[:,:,axis]
            qlow[:,axis]=np.minimum(qlow[:,axis],np.min(np.where(finite,values,np.inf),axis=1))
            qhigh[:,axis]=np.maximum(qhigh[:,axis],np.max(np.where(finite,values,-np.inf),axis=1))
        return result

    halves=[]
    for sign in (-1.,1.):
        q=initial.copy(); points=[q]
        for _ in range(steps):
            k1=direction(q); k2=direction(q+.5*sign*k1)
            k3=direction(q+.5*sign*k2); k4=direction(q+sign*k3)
            q=q+sign*(k1+2*k2+2*k3+k4)/6
            points.append(q)
        halves.append(points)
    primitive=np.stack(halves[0][:0:-1]+halves[1],axis=2).astype(np.float32)
    valid=np.isfinite(primitive).all(axis=(1,2,3))
    return primitive,valid,qlow,qhigh


def build(spec,config,input_root):
    import yaml
    import torch
    from experiments.Verify_Task4B_VelocityCurlMemorization import load_field
    from FMT_Utils.Task4B_CrossFlow_3D import dimensionless_primitive_features
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    out=Path(spec['output']); manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['config_sha256']==sha(config)
    if (out/'cache.npz').exists():
        raise RuntimeError('Frozen feature cache exists')
    old_spec=yaml.safe_load(Path(spec['field_source_config']).read_text())
    old_report=json.loads(Path('outputs/Verify_Task4B_VelocityCurlMemorization_4.1/build_summary.json').read_text())
    result=dict(version=spec['version'],config_sha256=sha(config),manifest_sha256=sha(out/'manifest.json'),
        flows={},**identity())
    chunks=[]; candidates=[]; offset=0
    for code,name in enumerate(('channel','tbl')):
        info=manifest['flows'][name]; source=Path(spec['labels_source'])/f'{name}_groundtruth.npz'
        assert sha(source)==info['label_sha256']
        with np.load(source) as d:
            labels=d['labels']; ids=d['vortex_ids']; low=d['domain_min_xyz']; high=d['domain_max_xyz']
            spacing=(high-low)/d['resolution_xyz']
        audit_manifest(ids,info['patches'],spec['split']['gap'])
        assert set(np.unique(ids[ids>0]))==set(old_report['flows'][name]['original_gt_ids'])
        flow=Path(input_root)/old_spec['flows'][name]['flow']
        assert sha(flow)==old_report['flows'][name]['flow_sha256']
        print('Loading',name,str(flow),flush=True)
        field,metadata,_,speed_rms=load_field(flow,name)
        axes=list(field.axes_zyx[::-1])
        if name=='channel':
            axes[0]=axes[0][:-1]; axes[1]=axes[1][:-1]
        rng=np.random.default_rng(spec['sampling']['seed']+code)
        per_patch=[]; candidate_parts=[]
        for p in info['patches']:
            flat=patch_candidates(labels,ids,p,rng,spec['sampling']['ordinary_train_cap_per_patch'])
            index=np.column_stack(np.unravel_index(flat,labels.shape))[:,::-1]
            points=low+(index+.5)*spacing
            plow=low+np.array(p['low'][::-1])*spacing
            phigh=low+np.array(p['high'][::-1])*spacing
            inner_low=np.array([a[min(np.searchsorted(a,l,'left'),len(a)-1)] for a,l in zip(axes,plow)])
            inner_high=np.array([a[max(np.searchsorted(a,u,'right')-1,0)] for a,u in zip(axes,phigh)])
            h=local_step(points,inner_low,inner_high,float(spacing.mean())*spec['streamlines']['base_step_mean_voxel_scale'],
                         spec['streamlines']['steps_per_direction'],spec['streamlines']['offset_over_step'])
            candidate_parts.append(dict(source=flat,patch=np.full(len(flat),p['patch_id'],np.int32),
                split=np.full(len(flat),p['split']=='test',np.int8),points=points,step=h,
                inner_low=np.broadcast_to(inner_low,points.shape).copy(),
                inner_high=np.broadcast_to(inner_high,points.shape).copy()))
            slices=tuple(slice(a,b) for a,b in zip(p['low'],p['high']))
            per_patch.append(dict(patch_id=p['patch_id'],split=p['split'],kind=p['kind'],instance_id=p['instance_id'],
                full_vortex_count=int(np.sum(labels[slices]>=0)),selected_count=len(flat),
                native_inner_low_xyz=inner_low.tolist(),native_inner_high_xyz=inner_high.tolist()))
        rows={k:np.concatenate([p[k] for p in candidate_parts]) for k in candidate_parts[0]}
        count=len(rows['source']); labels_rows=labels.ravel()[rows['source']]
        ids_rows=ids.ravel()[rows['source']]
        valid_initial=np.flatnonzero(rows['step']>spacing.mean()*1e-9)
        valid_all=np.zeros(count,bool); valid_features=[]; query_low=[];query_high=[];valid_indices=[]
        for start in range(0,len(valid_initial),spec['streamlines']['chunk_size']):
            selected=valid_initial[start:start+spec['streamlines']['chunk_size']]
            primitive,valid,ql,qh=integrate_local(field,rows['points'][selected],rows['step'][selected],
                rows['inner_low'][selected],rows['inner_high'][selected],spec['streamlines']['steps_per_direction'],
                spec['streamlines']['offset_over_step'],speed_rms*spec['streamlines']['minimum_speed_relative_rms'])
            if valid.any():
                _,fmt=dimensionless_primitive_features(primitive[valid],1.,**{k:spec['encoder'][k] for k in
                    ('num_freq','neighbor_scale','neighbor_pool','mode','include_chirality')})
                valid_features.append(fmt);valid_indices.append(selected[valid]);query_low.append(ql[valid]);query_high.append(qh[valid])
                valid_all[selected[valid]]=True
            if start==0 or start%20480==0:
                print(name,'integrated',min(start+len(selected),len(valid_initial)),'/',len(valid_initial),flush=True)
        selected=np.concatenate(valid_indices)
        chunks.append(dict(features=np.concatenate(valid_features),labels=labels_rows[selected],
            split=rows['split'][selected],volume=np.full(len(selected),code,np.int8),patch=rows['patch'][selected],
            source=rows['source'][selected],candidate_row=selected+offset,step=rows['step'][selected],
            query_low=np.concatenate(query_low),query_high=np.concatenate(query_high)))
        candidates.append(dict(labels=labels_rows,split=rows['split'],volume=np.full(count,code,np.int8),
            patch=rows['patch'],source=rows['source'],valid=valid_all,instance_id=ids_rows))
        offset+=count
        for p in per_patch:
            p['valid_count']=int(np.sum(valid_all & (rows['patch']==p['patch_id'])))
        result['flows'][name]=dict(patches=per_patch,metadata=metadata,flow_sha256=sha(flow),
            selected_count=count,valid_count=int(valid_all.sum()),invalid_count=int((~valid_all).sum()),
            physical_step_quantiles=np.quantile(rows['step'][selected],[0,.5,1]).tolist(),
            class_support={s:np.bincount(labels_rows[valid_all & (rows['split']==c)],minlength=4).tolist()
                           for s,c in [('train',0),('test',1)]})
        del field,primitive,rows,candidate_parts,labels,ids
    arrays={k:np.concatenate([p[k] for p in chunks]) for k in chunks[0]}
    candidate_arrays={k:np.concatenate([p[k] for p in candidates]) for k in candidates[0]}
    assert arrays['features'].shape[1]==spec['encoder']['expected_feature_dim']
    assert np.isfinite(arrays['features']).all()
    for code in (0,1):
        for split in (0,1):
            assert np.all(np.bincount(arrays['labels'][(arrays['volume']==code)&(arrays['split']==split)],minlength=4)>0)
    np.savez_compressed(out/'cache.npz',**arrays)
    np.savez_compressed(out/'candidates.npz',**candidate_arrays)
    result.update(cache_sha256=sha(out/'cache.npz'),candidates_sha256=sha(out/'candidates.npz'))
    write(out/'build_summary.json',result)
    print('Build complete',arrays['features'].shape,flush=True)


def metrics(y,pred):
    y=np.asarray(y);pred=np.asarray(pred)
    confusion=np.zeros((4,5),np.int64)
    np.add.at(confusion,(y,np.where(pred>=0,pred,4)),1)
    support=confusion.sum(1);tp=np.diag(confusion[:,:4]);fp=confusion[:,:4].sum(0)-tp;fn=support-tp
    iou=np.divide(tp,tp+fp+fn,out=np.zeros(4,float),where=(tp+fp+fn)>0)
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(4,float),where=(2*tp+fp+fn)>0)
    return dict(count=len(y),accuracy=float(tp.sum()/len(y)) if len(y) else None,
        mean_iou=float(iou.mean()) if len(y) else None,macro_f1=float(f1.mean()) if len(y) else None,
        per_class_iou=iou.tolist(),per_class_f1=f1.tolist(),class_support=support.tolist(),
        confusion_with_invalid_last_column=confusion.tolist(),invalid_prediction_count=int(np.sum(pred<0)))


def aggregate_unique(source,labels,probabilities,pred):
    unique,inverse=np.unique(source,return_inverse=True)
    sums=np.zeros((len(unique),4),np.float64);counts=np.zeros(len(unique),np.int64)
    valid=pred>=0
    np.add.at(sums,inverse[valid],probabilities[valid]);np.add.at(counts,inverse[valid],1)
    averaged=np.divide(sums,counts[:,None],out=np.zeros_like(sums),where=counts[:,None]>0)
    truth=np.full(len(unique),-1,np.int8);truth[inverse]=labels
    assert np.array_equal(truth[inverse],labels)
    predicted=np.where(counts>0,averaged.argmax(1),-2).astype(np.int8)
    return unique,truth,predicted,averaged.astype(np.float32)


def train(spec,config):
    import torch
    from FMT_Utils.Task4B_Classifier_3D import PathlineMulticlassClassifier3D,trainable_parameter_count
    from FMT_Utils.Task4A_PerVortexClustering_3D import fmt_base_feature_width
    out=Path(spec['output']);build_report=json.loads((out/'build_summary.json').read_text())
    assert build_report['config_sha256']==sha(config)
    assert sha(out/'cache.npz')==build_report['cache_sha256']
    assert sha(out/'candidates.npz')==build_report['candidates_sha256']
    if (out/'training_report.json').exists():
        raise RuntimeError('Frozen training report exists')
    with np.load(out/'cache.npz') as d:
        data={k:d[k] for k in d.files}
    with np.load(out/'candidates.npz') as d:
        candidates={k:d[k] for k in d.files}
    training=spec['training'];torch.manual_seed(training['seed']);np.random.seed(training['seed'])
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available(),'GPU required for this registered run'
    device=torch.device('cuda');fit=data['split']==0
    mean=data['features'][fit].mean(0,dtype=np.float64).astype(np.float32)
    std=np.maximum(data['features'][fit].std(0,dtype=np.float64),1e-6).astype(np.float32)
    features=(data['features']-mean)/std
    width=fmt_base_feature_width(spec['encoder']['num_freq'],spec['encoder']['mode'],spec['encoder']['include_chirality'])
    features[:,width:]*=spec['encoder']['neighbor_weight_after_train_standardization']
    np.savez_compressed(out/'normalization.npz',mean=mean,std=std,base_width=width)
    x=torch.from_numpy(features[fit]).to(device);y=torch.from_numpy(data['labels'][fit].astype(np.int64)).to(device)
    support=np.bincount(data['labels'][fit],minlength=4)
    weight=1/np.sqrt(support);weight/=weight.mean()
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weight,dtype=torch.float32,device=device))
    model=PathlineMulticlassClassifier3D(fmt_dim=features.shape[1],num_classes=4,**spec['model']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=training['learning_rate'],weight_decay=training['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,training['epochs'],eta_min=training['minimum_learning_rate'])
    started=time.perf_counter()
    with (out/'training_history.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['epoch','train_loss','train_accuracy_during_updates','learning_rate']);writer.writeheader()
        for epoch in range(1,training['epochs']+1):
            model.train();order=np.random.default_rng(training['seed']+epoch).permutation(len(y))
            loss_sum=0.;correct=0
            for start in range(0,len(order),training['batch_size']):
                index=torch.from_numpy(order[start:start+training['batch_size']]).to(device)
                optimizer.zero_grad(set_to_none=True);logits=model(None,x[index]);loss=criterion(logits,y[index])
                loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(index)
                correct+=int((logits.argmax(1)==y[index]).sum())
            row=dict(epoch=epoch,train_loss=loss_sum/len(y),train_accuracy_during_updates=correct/len(y),learning_rate=optimizer.param_groups[0]['lr'])
            writer.writerow(row);handle.flush();scheduler.step()
            if epoch<=3 or epoch%10==0: print(json.dumps(row),flush=True)
    # Exactly one final held-out inference; no validation/test model selection.
    model.eval();probability=[]
    with torch.no_grad():
        for start in range(0,len(features),4096):
            logits=model(None,torch.from_numpy(features[start:start+4096]).to(device))
            probability.append(logits.softmax(1).cpu().numpy())
    probability=np.concatenate(probability)
    full_probability=np.zeros((len(candidates['labels']),4),np.float32)
    full_pred=np.full(len(candidates['labels']),-2,np.int8)
    full_probability[data['candidate_row']]=probability
    full_pred[data['candidate_row']]=probability.argmax(1)
    np.savez_compressed(out/'predictions.npz',predictions=full_pred,probabilities=full_probability,**candidates)
    report=dict(version=spec['version'],config_sha256=sha(config),build_sha256=sha(out/'build_summary.json'),
        cache_sha256=sha(out/'cache.npz'),predictions_sha256=sha(out/'predictions.npz'),**identity(),
        device=torch.cuda.get_device_name(),torch_version=torch.__version__,epochs=training['epochs'],
        model_selection='fixed_final_epoch',test_inference_passes=1,parameter_count=trainable_parameter_count(model),
        elapsed_seconds=time.perf_counter()-started,class_weights=weight.tolist(),flows={})
    per_patch=[]
    for code,name in enumerate(('channel','tbl')):
        f={}
        for split_code,split_name in ((0,'train_sampled'),(1,'test_dense')):
            mask=(candidates['volume']==code)&(candidates['split']==split_code)
            unique,truth,pred,probs=aggregate_unique(candidates['source'][mask],candidates['labels'][mask],full_probability[mask],full_pred[mask])
            f[split_name]=dict(patch_occurrences_including_invalid=metrics(candidates['labels'][mask],full_pred[mask]),
                unique_voxels_including_invalid=metrics(truth,pred),
                unique_voxels_valid_only=metrics(truth[pred>=0],pred[pred>=0]))
            if split_code==1:
                np.savez_compressed(out/f'{name}_test_segmentation.npz',source=unique,labels=truth,predictions=pred,probabilities=probs)
        for p in build_report['flows'][name]['patches']:
            mask=(candidates['volume']==code)&(candidates['patch']==p['patch_id'])
            if p['split']=='test':
                item=metrics(candidates['labels'][mask],full_pred[mask]);per_patch.append(dict(flow=name,patch_id=p['patch_id'],**item))
        f['test_patch_mean_iou']=float(np.mean([p['mean_iou'] for p in per_patch if p['flow']==name and p['count']]))
        report['flows'][name]=f
    write(out/'test_patch_metrics.json',per_patch)
    write(out/'training_report.json',report)
    print(json.dumps(report),flush=True)


def audit(spec,config):
    out=Path(spec['output']);manifest=json.loads((out/'manifest.json').read_text())
    build=json.loads((out/'build_summary.json').read_text());report=json.loads((out/'training_report.json').read_text())
    assert manifest['config_sha256']==build['config_sha256']==report['config_sha256']==sha(config)
    assert build['manifest_sha256']==sha(out/'manifest.json')
    assert build['cache_sha256']==report['cache_sha256']==sha(out/'cache.npz')
    assert build['candidates_sha256']==sha(out/'candidates.npz')
    assert report['predictions_sha256']==sha(out/'predictions.npz')
    with np.load(out/'cache.npz') as d: cache={k:d[k] for k in d.files}
    with np.load(out/'predictions.npz') as d: predictions={k:d[k] for k in d.files}
    with np.load(out/'normalization.npz') as d:
        np.testing.assert_allclose(d['mean'],cache['features'][cache['split']==0].mean(0,dtype=np.float64).astype(np.float32),rtol=0,atol=0)
        np.testing.assert_allclose(d['std'],np.maximum(cache['features'][cache['split']==0].std(0,dtype=np.float64),1e-6).astype(np.float32),rtol=0,atol=0)
    assert np.array_equal(np.flatnonzero(predictions['valid']),cache['candidate_row'])
    assert np.array_equal(cache['labels'],predictions['labels'][cache['candidate_row']])
    assert np.array_equal(cache['split'],predictions['split'][cache['candidate_row']])
    np.testing.assert_allclose(predictions['probabilities'][predictions['valid']].sum(1),1.,atol=1e-5)
    assert np.all(predictions['predictions'][~predictions['valid']]==-2)
    checks={}
    for code,name in enumerate(('channel','tbl')):
        source=Path(spec['labels_source'])/f'{name}_groundtruth.npz'
        assert sha(source)==manifest['flows'][name]['label_sha256']
        with np.load(source) as d:
            split_audit=audit_manifest(d['vortex_ids'],manifest['flows'][name]['patches'],spec['split']['gap'])
            rows=predictions['volume']==code
            assert np.array_equal(predictions['labels'][rows],d['labels'].ravel()[predictions['source'][rows]])
            assert np.array_equal(predictions['instance_id'][rows],d['vortex_ids'].ravel()[predictions['source'][rows]])
            for p in manifest['flows'][name]['patches']:
                selected=rows&(predictions['patch']==p['patch_id'])
                index=np.column_stack(np.unravel_index(predictions['source'][selected],d['labels'].shape))
                assert np.all(index>=p['low']) and np.all(index< p['high'])
                if p['split']=='test':
                    slices=tuple(slice(a,b) for a,b in zip(p['low'],p['high']))
                    assert len(index)==np.sum(d['labels'][slices]>=0),'Test patch is not dense'
                valid_rows=(cache['volume']==code)&(cache['patch']==p['patch_id'])
                bounds=build['flows'][name]['patches'][p['patch_id']]
                assert np.all(cache['query_low'][valid_rows]>=np.array(bounds['native_inner_low_xyz'])-1e-11)
                assert np.all(cache['query_high'][valid_rows]<=np.array(bounds['native_inner_high_xyz'])+1e-11)
            train_sources=set(predictions['source'][rows&(predictions['split']==0)])
            test_sources=set(predictions['source'][rows&(predictions['split']==1)])
            assert train_sources.isdisjoint(test_sources)
        with np.load(out/f'{name}_test_segmentation.npz') as d:
            mask=(predictions['volume']==code)&(predictions['split']==1)
            u,y,p,probs=aggregate_unique(predictions['source'][mask],predictions['labels'][mask],
                predictions['probabilities'][mask],predictions['predictions'][mask])
            assert np.array_equal(d['source'],u) and np.array_equal(d['labels'],y) and np.array_equal(d['predictions'],p)
            np.testing.assert_allclose(d['probabilities'],probs,rtol=0,atol=0)
            measured=metrics(d['labels'],d['predictions'])
            assert measured==report['flows'][name]['test_dense']['unique_voxels_including_invalid']
        checks[name]=dict(split=split_audit,unique_source_overlap=0,dense_test_coverage_checked=True,primitive_query_bounds_checked=True)
    assert not any(out.rglob('*.pt')) and not any(out.rglob('*.pth')) and not any(out.rglob('*.ckpt'))
    assert report['epochs']==spec['training']['epochs'] and report['test_inference_passes']==1
    write(out/'audit.json',dict(passed=True,checks=checks,normalization_train_only=True,no_checkpoint=True,**identity()))
    print('AUDIT PASS',flush=True)


def render_predictions(spec):
    from experiments.Visualize_Task4B_VelocityCurl_3D import render
    out=Path(spec['output']);folder=out/'figures_3d';folder.mkdir(exist_ok=True)
    manifest=json.loads((out/'manifest.json').read_text())
    with np.load(out/'predictions.npz') as d: pred={k:d[k] for k in d.files}
    rendered=[]
    for code,name in enumerate(('channel','tbl')):
        info=manifest['flows'][name];low=np.array(info['low_xyz']);high=np.array(info['high_xyz'])
        resolution=np.array(info['resolution_xyz']);spacing=(high-low)/resolution
        chosen=[p for p in info['patches'] if p['split']=='test' and p['kind']=='hairpin_bbox'][:2]
        for p in chosen:
            select=(pred['volume']==code)&(pred['patch']==p['patch_id'])
            indices=np.column_stack(np.unravel_index(pred['source'][select],tuple(resolution[::-1])))[:,::-1]
            points=low+(indices+.5)*spacing
            plow=low+np.array(p['low'][::-1])*spacing;phigh=low+np.array(p['high'][::-1])*spacing
            for key,title in [('labels','Proxy GT'),('predictions','FMT prediction')]:
                values=pred[key][select];valid=values>=0
                if not valid.any(): continue
                file=folder/f'{name}_patch{p["patch_id"]}_{key}.png'
                rendered.append(render(points[valid],values[valid],spacing,plow,phigh,file,
                    f'{name.upper()} | Held-out patch {p["patch_id"]} | {title}',
                    f'Instance {p["instance_id"]} | {len(points):,} vortex voxels | invalid primitives {int((~pred["valid"][select]).sum())} | fixed epoch {spec["training"]["epochs"]}'))
    write(folder/'render_summary.json',rendered)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','build','train','audit','render'])
    parser.add_argument('--config',default='config/mainExp_Task4B_PatchSegmentation_5.1.json')
    parser.add_argument('--input-root',default='inputs')
    args=parser.parse_args();spec=json.loads(Path(args.config).read_text())
    if args.stage=='plan':plan(spec,args.config)
    elif args.stage=='build':build(spec,args.config,args.input_root)
    elif args.stage=='train':train(spec,args.config)
    elif args.stage=='audit':audit(spec,args.config)
    else:render_predictions(spec)
