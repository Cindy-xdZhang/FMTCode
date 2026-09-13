"""Train-only tuning of geometry-derived curl/FMT classifiers, version 5.2."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from FMT_Utils.Task4B_GeometricCurl_3D import internal_split,integrate_geometry,encoded_geometry
from FMT_Utils.Task4B_PatchSplit_3D import local_step,overlaps
from experiments.Task4B_PatchSegmentation_5_1 import metrics,aggregate_unique


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def identity():
    return dict(host=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),
        array_id=os.environ.get('SLURM_ARRAY_TASK_ID'),script_sha256=sha(__file__),
        geometry_sha256=sha('FMT_Utils/Task4B_GeometricCurl_3D.py'))


def load_npz(path):
    with np.load(path) as d:return {k:d[k] for k in d.files}


def plan(spec,config):
    out=Path(spec['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'inner_split.json').exists():raise RuntimeError('Frozen inner split exists')
    manifest=json.loads((Path(spec['baseline'])/'manifest.json').read_text())
    split=internal_split(manifest,**spec['inner_split'])
    write(out/'inner_split.json',dict(config_sha256=sha(config),baseline_manifest_sha256=sha(Path(spec['baseline'])/'manifest.json'),flows=split))
    (out/'config_snapshot.json').write_bytes(Path(config).read_bytes())
    print({k:{q:v for q,v in d.items() if q!='patch_roles'} for k,d in split.items()},flush=True)


def generate_features(spec,base,selected,input_root,modes):
    import torch
    import yaml
    from experiments.Verify_Task4B_VelocityCurlMemorization import load_field
    baseline=Path(spec['baseline']);manifest=json.loads((baseline/'manifest.json').read_text())
    build=json.loads((baseline/'build_summary.json').read_text())
    field_spec=yaml.safe_load(Path('config/Verify_Task4B_VelocityCurlMemorization_4.1.yaml').read_text())
    results={mode:[] for mode in modes};cosines={mode:[] for mode in modes};diagnostics={}
    for code,name in enumerate(('channel','tbl')):
        rows=selected[base['volume'][selected]==code]
        source=Path(input_root)/field_spec['flows'][name]['flow']
        assert sha(source)==build['flows'][name]['flow_sha256']
        field,_,_,_=load_field(source,name)
        # Keep interpolation arithmetic in float64; stored native data remain float32.
        interpolator=field._interpolator if name=='channel' else field._velocity_interpolator
        def velocity(points):return interpolator(points[:,[2,1,0]])
        info=manifest['flows'][name];domain_low=np.array(info['low_xyz']);domain_high=np.array(info['high_xyz'])
        resolution=np.array(info['resolution_xyz']);spacing=(domain_high-domain_low)/resolution
        index=np.column_stack(np.unravel_index(base['source'][rows],tuple(resolution[::-1])))[:,::-1]
        seeds=domain_low+(index+.5)*spacing
        bounds={p['patch_id']:p for p in build['flows'][name]['patches']}
        lower=np.array([bounds[int(i)]['native_inner_low_xyz'] for i in base['patch'][rows]])
        upper=np.array([bounds[int(i)]['native_inner_high_xyz'] for i in base['patch'][rows]])
        scales={}
        for pid in np.unique(base['patch'][rows]):
            p=bounds[int(pid)]
            starts=[np.searchsorted(a,l,'left') for a,l in zip(field.axes_zyx,p['native_inner_low_xyz'][::-1])]
            stops=[np.searchsorted(a,h,'right') for a,h in zip(field.axes_zyx,p['native_inner_high_xyz'][::-1])]
            values=field.velocity_zyx3[tuple(slice(a,b) for a,b in zip(starts,stops))]
            scales[int(pid)]=float(np.linalg.norm(values.astype(np.float64),axis=-1).max())*(1+1e-10)
        scale=np.array([scales[int(pid)] for pid in base['patch'][rows]])
        step=local_step(seeds,lower,upper,float(spacing.mean())*.5,spec['geometry']['steps'],spec['geometry']['offset'])
        assert (step>0).all() and (scale>0).all()
        for mode in modes:
            features=[];cos=[]
            for start in range(0,len(rows),spec['geometry']['chunk_size']):
                stop=min(start+spec['geometry']['chunk_size'],len(rows))
                q,ql,qh=integrate_geometry(velocity,seeds[start:stop],step[start:stop],lower[start:stop],upper[start:stop],
                    scale[start:stop],mode=mode,steps=spec['geometry']['steps'],offset=spec['geometry']['offset'])
                assert np.all(ql>=lower[start:stop]-1e-11) and np.all(qh<=upper[start:stop]+1e-11)
                x,c=encoded_geometry(q);features.append(x);cos.append(c)
                if start%20480==0:print(name,mode,start,'/',len(rows),flush=True)
            results[mode].append(np.concatenate(features));cosines[mode].append(np.concatenate(cos))
        diagnostics[name]=dict(rows=len(rows),step_quantiles=np.quantile(step,[0,.5,1]).tolist(),
            patch_speed_scale_quantiles=np.quantile(scale,[0,.5,1]).tolist(),native_queries_inside_patch=True)
        del field
    return {k:np.concatenate(v) for k,v in results.items()},{k:np.concatenate(v) for k,v in cosines.items()},diagnostics


def build(spec,config,input_root):
    import torch
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','8')))
    out=Path(spec['output']);baseline=Path(spec['baseline'])
    split=json.loads((out/'inner_split.json').read_text())
    assert split['config_sha256']==sha(config)
    old_build=json.loads((baseline/'build_summary.json').read_text())
    assert sha(baseline/'cache.npz')==old_build['cache_sha256']
    if (out/'train_features.npz').exists():raise RuntimeError('Frozen training features exist')
    base=load_npz(baseline/'cache.npz');selected=np.flatnonzero(base['split']==0)
    assert np.all(np.diff(base['volume'][selected])>=0)
    features,cosines,diagnostics=generate_features(spec,base,selected,input_root,['unit','time'])
    names=['channel','tbl']
    roles=np.array([split['flows'][names[int(v)]]['patch_roles'][str(int(p))]
                    for v,p in zip(base['volume'][selected],base['patch'][selected])],np.int8)
    arrays=dict(baseline=base['features'][selected],unit=features['unit'],time=features['time'],
        labels=base['labels'][selected],volume=base['volume'][selected],patch=base['patch'][selected],
        source=base['source'][selected],baseline_row=selected,role=roles)
    for code,name in enumerate(names):
        diagnostics[name]['orientation_agreement_fit_only']={}
        for mode in cosines:
            use=(arrays['volume']==code)&(roles==0)
            truth=np.isin(arrays['labels'][use],[0,3])
            diagnostics[name]['orientation_agreement_fit_only'][mode]=float(np.mean((cosines[mode][use]**2>=.5)==truth))
        for role in (0,1):
            assert np.all(np.bincount(arrays['labels'][(arrays['volume']==code)&(roles==role)],minlength=4)>0)
    # Common balanced finite-sample memorization diagnostic, drawn only from fit.
    rng=np.random.default_rng(spec['memorization']['seed']);subset=[]
    for code in (0,1):
        for k in range(4):
            rows=np.flatnonzero((roles==0)&(arrays['volume']==code)&(arrays['labels']==k))
            subset.extend(rng.choice(rows,min(len(rows),spec['memorization']['per_flow_per_class']),replace=False))
    arrays['memorization_rows']=np.sort(subset)
    np.savez_compressed(out/'train_features.npz',**arrays)
    write(out/'build_summary.json',dict(version=spec['version'],config_sha256=sha(config),
        inner_split_sha256=sha(out/'inner_split.json'),baseline_cache_sha256=sha(baseline/'cache.npz'),
        training_features_sha256=sha(out/'train_features.npz'),row_count=len(roles),
        feature_dimensions={k:int(arrays[k].shape[1]) for k in ('baseline','unit','time')},
        role_counts={str(k):int(np.sum(roles==k)) for k in (-1,0,1)},
        memorization_count=len(subset),flows=diagnostics,original_test_rows_used=0,**identity()))


def make_model(candidate,dimension,diagnostic=False):
    import torch
    from torch import nn
    width=candidate['width'];drop=0 if diagnostic else candidate['dropout']
    if candidate['architecture']=='mlp':
        return nn.Sequential(nn.Linear(dimension,width),nn.LayerNorm(width),nn.GELU(),
            nn.Linear(width,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(drop),nn.Linear(width,4))
    class ResidualClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.input=nn.Sequential(nn.Linear(dimension,width),nn.LayerNorm(width),nn.GELU())
            self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.GELU(),
                nn.Dropout(drop),nn.Linear(width,width)) for _ in range(candidate['depth'])])
            self.final_norm=nn.LayerNorm(width)
            self.head=nn.Linear(width,4)
        def forward(self,x):
            x=self.input(x)
            for block in self.blocks:x=x+block(x)
            logits=self.head(self.final_norm(x))
            if candidate['architecture']=='factorized':
                membership=logits[:,:2].log_softmax(1);orientation=logits[:,2:].log_softmax(1)
                logits=torch.stack((membership[:,0]+orientation[:,0],membership[:,0]+orientation[:,1],
                    membership[:,1]+orientation[:,1],membership[:,1]+orientation[:,0]),1)
            return logits
    return ResidualClassifier()


def normalize(x,fit):
    mean=x[fit].mean(0,dtype=np.float64).astype(np.float32)
    std=np.maximum(x[fit].std(0,dtype=np.float64),1e-6).astype(np.float32)
    return apply_normalization(x,mean,std),mean,std


def apply_normalization(x,mean,std):
    value=(x-mean)/std
    value[:,23:161]*=.5
    if x.shape[1]>161:value[:,184:322]*=.5
    assert np.isfinite(value).all()
    return value


def probabilities(model,x,batch=4096):
    import torch
    model.eval();pieces=[]
    with torch.no_grad():
        for start in range(0,len(x),batch):pieces.append(model(x[start:start+batch]).softmax(1).cpu().numpy())
    return np.concatenate(pieces)


def flow_metrics(data,indices,p):
    result={}
    for code,name in enumerate(('channel','tbl')):
        use=data['volume'][indices]==code
        source=data['source'][indices][use];labels=data['labels'][indices][use]
        _,y,pred,_=aggregate_unique(source,labels,p[use],p[use].argmax(1))
        result[name]=metrics(y,pred)
    return result


def optimize(spec,candidate,data,indices,validation=None,diagnostic=False,epochs=None,history=None):
    import torch
    training=spec['memorization'] if diagnostic else spec['training']
    torch.manual_seed(training['seed']);torch.cuda.manual_seed_all(training['seed'])
    x_np,mean,std=normalize(data[candidate['representation']],indices)
    x=torch.from_numpy(x_np[indices]).cuda();y=torch.from_numpy(data['labels'][indices].astype(np.int64)).cuda()
    val_x=torch.from_numpy(x_np[validation]).cuda() if validation is not None else None
    model=make_model(candidate,x.shape[1],diagnostic).cuda()
    support=np.bincount(data['labels'][indices],minlength=4);weight=1/np.sqrt(support);weight/=weight.mean()
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weight,dtype=torch.float32,device='cuda'))
    optimizer=torch.optim.Adam(model.parameters(),lr=candidate['lr'],weight_decay=0 if diagnostic else candidate['weight_decay'])
    total=epochs or training['epochs']
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,training['epochs'],eta_min=spec['training']['minimum_learning_rate'])
    best_score=-1.;best=None;best_probability=None;streak=0;passed=False;min_errors=len(y);min_error_epoch=0
    log=[];started=time.perf_counter()
    for epoch in range(1,total+1):
        model.train();order=np.random.default_rng(training['seed']+epoch).permutation(len(y))
        loss_sum=0.;correct=0
        for start in range(0,len(order),training['batch_size']):
            index=torch.from_numpy(order[start:start+training['batch_size']]).cuda()
            optimizer.zero_grad(set_to_none=True);logits=model(x[index]);loss=criterion(logits,y[index])
            loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(index);correct+=int((logits.argmax(1)==y[index]).sum())
        row=dict(epoch=epoch,loss=loss_sum/len(y),update_accuracy=correct/len(y),learning_rate=optimizer.param_groups[0]['lr'])
        scheduler.step()
        if diagnostic:
            p=probabilities(model,x);errors=int(np.sum(p.argmax(1)!=data['labels'][indices]))
            row['fit_error_count']=errors;row['fit_accuracy']=1-errors/len(indices)
            if errors<min_errors:min_errors=errors;min_error_epoch=epoch
            streak=streak+1 if errors==0 else 0
            if streak>=training['required_zero_error_epochs']:passed=True
        elif validation is not None and (epoch==1 or epoch%training['evaluation_interval']==0 or epoch==total):
            p=probabilities(model,val_x);measured=flow_metrics(data,validation,p)
            score=float(np.mean([d['mean_iou'] for d in measured.values()]));row['validation_mean_iou']=score
            if score>best_score:
                best_score=score;best=dict(epoch=epoch,score=score,flows=measured);best_probability=p.copy()
        log.append(row)
        if epoch<=3 or epoch%50==0 or passed:print(candidate['name'],'memorization' if diagnostic else 'training',json.dumps(row),flush=True)
        if passed:break
    if history:
        keys=list(dict.fromkeys(k for r in log for k in r))
        with Path(history).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(log)
    summary=dict(epochs_completed=epoch,seconds=time.perf_counter()-started,parameter_count=sum(p.numel() for p in model.parameters()),
        fit_rows=len(indices),class_weights=weight.tolist(),best_validation=best,
        memorization_passed=passed if diagnostic else None,minimum_fit_errors=min_errors if diagnostic else None,
        minimum_error_epoch=min_error_epoch if diagnostic else None)
    return model,mean,std,summary,best_probability


def search(spec,config,index):
    import torch
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','8')))
    assert torch.cuda.is_available()
    out=Path(spec['output']);build=json.loads((out/'build_summary.json').read_text())
    assert build['config_sha256']==sha(config) and sha(out/'train_features.npz')==build['training_features_sha256']
    data=load_npz(out/'train_features.npz');candidate=spec['candidates'][index]
    dest=out/'search'/candidate['name'];dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():raise RuntimeError('Frozen candidate result exists')
    assert np.all(data['role'][data['memorization_rows']]==0)
    model,_,_,memorization,_=optimize(spec,candidate,data,data['memorization_rows'],diagnostic=True,history=dest/'memorization.csv')
    del model;torch.cuda.empty_cache()
    fit=np.flatnonzero(data['role']==0);validation=np.flatnonzero(data['role']==1)
    model,mean,std,measured,p=optimize(spec,candidate,data,fit,validation,history=dest/'training.csv')
    np.savez_compressed(dest/'validation_predictions.npz',rows=validation,probabilities=p,
        labels=data['labels'][validation],source=data['source'][validation],volume=data['volume'][validation],mean=mean,std=std)
    write(dest/'summary.json',dict(candidate_index=index,candidate=candidate,config_sha256=sha(config),
        training_features_sha256=build['training_features_sha256'],memorization=memorization,training=measured,
        predictions_sha256=sha(dest/'validation_predictions.npz'),device=torch.cuda.get_device_name(),
        original_test_used=False,**identity()))
    print('Candidate complete',candidate['name'],measured['best_validation'],flush=True)


def select(spec,config):
    out=Path(spec['output'])
    if (out/'selection.json').exists():raise RuntimeError('Frozen selection exists')
    results=[]
    for index,candidate in enumerate(spec['candidates']):
        folder=out/'search'/candidate['name'];r=json.loads((folder/'summary.json').read_text())
        assert r['config_sha256']==sha(config) and not r['original_test_used']
        assert r['predictions_sha256']==sha(folder/'validation_predictions.npz')
        d=load_npz(folder/'validation_predictions.npz')
        measured=flow_metrics(d,np.arange(len(d['labels'])),d['probabilities'])
        assert measured==r['training']['best_validation']['flows']
        score=float(np.mean([v['mean_iou'] for v in measured.values()]))
        assert score==r['training']['best_validation']['score']
        results.append(dict(index=index,name=candidate['name'],score=score,
            epoch=r['training']['best_validation']['epoch'],memorization=r['memorization'],flows=measured,
            summary_sha256=sha(folder/'summary.json')))
    winner=max(results,key=lambda r:(r['score'],-r['index']))
    write(out/'selection.json',dict(config_sha256=sha(config),winner=winner,candidates=results,
        selection_metric=spec['training']['selection'],test_used_for_selection=False,**identity()))
    print('Frozen winner',winner,flush=True)


def final(spec,config,input_root):
    import torch
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','8')))
    assert torch.cuda.is_available()
    out=Path(spec['output']);selection=json.loads((out/'selection.json').read_text())
    assert selection['config_sha256']==sha(config) and not selection['test_used_for_selection']
    if (out/'final_report.json').exists():raise RuntimeError('Frozen final result exists')
    chosen=selection['winner'];candidate=spec['candidates'][chosen['index']]
    data=load_npz(out/'train_features.npz');fit=np.arange(len(data['labels']))
    # Epoch selection was made on a longer cosine schedule. Refit follows the
    # same schedule prefix, not a newly shortened annealing curve.
    model,mean,std,trained,_=optimize(spec,candidate,data,fit,epochs=chosen['epoch'],history=out/'final_training.csv')
    baseline=Path(spec['baseline']);base=load_npz(baseline/'cache.npz');rows=np.flatnonzero(base['split']==1)
    if candidate['representation']=='baseline':test_features=base['features'][rows];geometry={}
    else:
        arrays,_,geometry=generate_features(spec,base,rows,input_root,[candidate['representation']])
        test_features=arrays[candidate['representation']]
    test_x=torch.from_numpy(apply_normalization(test_features,mean,std)).cuda()
    prob=probabilities(model,test_x)
    candidates=load_npz(baseline/'candidates.npz');full=np.flatnonzero(candidates['split']==1)
    mapping=np.full(len(candidates['labels']),-1,np.int64);mapping[full]=np.arange(len(full))
    dest=mapping[base['candidate_row'][rows]];assert np.all(dest>=0)
    all_prob=np.zeros((len(full),4),np.float32);pred=np.full(len(full),-2,np.int8)
    all_prob[dest]=prob;pred[dest]=prob.argmax(1)
    arrays={k:v[full] for k,v in candidates.items()};arrays.update(predictions=pred,probabilities=all_prob)
    np.savez_compressed(out/'final_test_predictions.npz',**arrays)
    result={}
    for code,name in enumerate(('channel','tbl')):
        use=arrays['volume']==code
        unique,y,p,probs=aggregate_unique(arrays['source'][use],arrays['labels'][use],all_prob[use],pred[use])
        np.savez_compressed(out/f'{name}_test_segmentation.npz',source=unique,labels=y,predictions=p,probabilities=probs)
        result[name]=dict(unique_voxels_including_invalid=metrics(y,p),valid_coverage=float(np.mean(p>=0)),
            orientation_accuracy=float(np.mean((p>=0)&(np.isin(y,[0,3])==np.isin(p,[0,3])))),
            hairpin_membership_accuracy=float(np.mean((p>=0)&((y>=2)==(p>=2)))))
    train_p=probabilities(model,torch.from_numpy(apply_normalization(data[candidate['representation']],mean,std)).cuda())
    write(out/'final_report.json',dict(version=spec['version'],config_sha256=sha(config),selection_sha256=sha(out/'selection.json'),
        selected_candidate=candidate,selected_epochs=chosen['epoch'],training=trained,
        training_unique_voxels=flow_metrics(data,fit,train_p),flows=result,
        prediction_sha256=sha(out/'final_test_predictions.npz'),test_inference_passes=1,
        reused_5p1_test_not_fresh_confirmation=True,test_used_for_selection=False,
        geometry=geometry,device=torch.cuda.get_device_name(),**identity()))
    print('Final test complete',result,flush=True)


def audit(spec,config):
    out=Path(spec['output']);baseline=Path(spec['baseline'])
    split=json.loads((out/'inner_split.json').read_text());manifest=json.loads((baseline/'manifest.json').read_text())
    expected=internal_split(manifest,**spec['inner_split'])
    assert json.loads(json.dumps(expected))==split['flows']
    data=load_npz(out/'train_features.npz');old=load_npz(baseline/'cache.npz')
    assert np.all(old['split'][data['baseline_row']]==0)
    np.testing.assert_array_equal(data['baseline'],old['features'][data['baseline_row']])
    for code,name in enumerate(('channel','tbl')):
        roles=split['flows'][name]['patch_roles']
        for i in np.flatnonzero(data['volume']==code):assert data['role'][i]==roles[str(int(data['patch'][i]))]
        train_sources=set(data['source'][(data['volume']==code)&(data['role']==0)])
        val_sources=set(data['source'][(data['volume']==code)&(data['role']==1)])
        assert train_sources.isdisjoint(val_sources)
    selection=json.loads((out/'selection.json').read_text());report=json.loads((out/'final_report.json').read_text())
    assert report['selection_sha256']==sha(out/'selection.json')
    assert report['selected_epochs']==selection['winner']['epoch'] and not report['test_used_for_selection']
    assert report['prediction_sha256']==sha(out/'final_test_predictions.npz')
    p=load_npz(out/'final_test_predictions.npz');c=load_npz(baseline/'candidates.npz');test=c['split']==1
    for k in ('labels','source','patch','volume','valid'):np.testing.assert_array_equal(p[k],c[k][test])
    for code,name in enumerate(('channel','tbl')):
        use=p['volume']==code
        _,y,pred,_=aggregate_unique(p['source'][use],p['labels'][use],p['probabilities'][use],p['predictions'][use])
        assert metrics(y,pred)==report['flows'][name]['unique_voxels_including_invalid']
    assert not any(out.rglob('*.pt')) and not any(out.rglob('*.pth')) and not any(out.rglob('*.ckpt'))
    write(out/'audit.json',dict(passed=True,fit_validation_source_overlap=0,original_test_used_for_selection=False,
        test_support_same_as_5p1=True,no_checkpoints=True,**identity()))
    print('AUDIT PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['plan','build','search','select','final','audit'])
    p.add_argument('--config',default='config/Other_Task4B_GeometrySearch_5.2.json');p.add_argument('--index',type=int,default=0)
    p.add_argument('--input-root',default='inputs');args=p.parse_args();spec=json.loads(Path(args.config).read_text())
    if args.stage=='plan':plan(spec,args.config)
    elif args.stage=='build':build(spec,args.config,args.input_root)
    elif args.stage=='search':search(spec,args.config,args.index)
    elif args.stage=='select':select(spec,args.config)
    elif args.stage=='final':final(spec,args.config,args.input_root)
    else:audit(spec,args.config)
