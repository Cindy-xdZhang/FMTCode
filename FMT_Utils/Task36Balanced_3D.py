"""Version 1.2: radius-unit reconstruction loss and task-metric checkpoint selection."""
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from FMT_Utils.Task36MultiGate_3D import (
    Task36MultiGate, task_flags, balanced_loss, choose_threshold,
    evaluate_cached as evaluate_legacy, predict, classification_metrics)
from FMT_Utils.Task6FMTGeometryMoE_3D import cached, fit_statistics
from FMT_Utils.Task6DirectNeural_3D import check_geometry
from FMT_Utils.Task6PNNSchedule_3D import LearningRateSchedule
from FMT_Utils.FlowMapData_3D import write_json


def task_metric_score(average_precision, rmse_r):
    """Joint selection uses the stated task metrics, never train geometry variance."""
    values = ([1. - average_precision] if average_precision is not None else [])
    values += [rmse_r] if rmse_r is not None else []
    assert values and np.isfinite(values).all()
    return sum(values)


def evaluate_cached(*args, **kwargs):
    metrics, probability = evaluate_legacy(*args, **kwargs)
    ap = metrics['task3']['average_precision'] if metrics['task3'] is not None else None
    metrics['legacy_selection_score_diagnostic'] = metrics['selection_score']
    metrics['selection_score'] = task_metric_score(ap, metrics['task6_rmse_r'])
    metrics['physical_reconstruction_mse_r2'] = metrics['task6_rmse_r']**2 if metrics['task6_rmse_r'] is not None else None
    return metrics, probability


def build_balanced_model(statistics, candidate, variant):
    construction_variant = 'joint_raw_task_gates' if variant == 'joint_raw_shared_gate' else variant
    model = Task36MultiGate(statistics, candidate, construction_variant)
    if variant == 'joint_raw_shared_gate':
        # Keep all allocated layers and their initialization identical to the
        # FMT shared-gate arm. Only replace the A input by raw geometry.
        model.variant = 'joint_shared_gate'
    model.experimental_variant = variant
    return model


def train_balanced(y,labels,vy,vlabels,variant,candidate,training,seed,device,folder):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    check_geometry(y)
    check_geometry(vy)
    assert len(labels)==len(y) and len(vlabels)==len(vy)
    assert set(np.unique(labels))=={0,1} and set(np.unique(vlabels))=={0,1}
    torch.manual_seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(seed)
    model=build_balanced_model(fit_statistics(y),candidate,variant).to(device)
    features=cached(model,y,device)
    model.condition(features)
    vfeatures=cached(model,vy,device)
    target=torch.as_tensor(y,device=device)
    labs=torch.as_tensor(labels,dtype=torch.float32,device=device)
    positive=float(np.mean(labels))
    use3,use6=task_flags(variant)
    optimizer=torch.optim.AdamW(model.parameters(),lr=candidate['learning_rate'],weight_decay=candidate['weight_decay'])
    schedule=LearningRateSchedule(optimizer,candidate,training)
    batch=min(training['batch_size'],len(y))
    batches=math.ceil(len(y)/batch)
    rng=np.random.default_rng(seed)
    history=[]
    best,best_step,best_state=float('inf'),None,None
    start=time.monotonic()

    def record(step):
        nonlocal best,best_step,best_state
        train,_=evaluate_cached(model,features,y,labels,positive)
        val,_=evaluate_cached(model,vfeatures,vy,vlabels,positive)
        score=val['selection_score']
        assert np.isfinite(score)
        if step>0 and score<best:
            best,best_step=score,step
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        row=dict(step=step,train=train,validation=val,seconds=time.monotonic()-start,learning_rate=schedule.last_used)
        history.append(row)
        write_json(folder/'progress.json',row)
        print(dict(variant=variant,**row),flush=True)

    record(0)
    for step in range(training['updates']):
        if step%batches==0:
            order=rng.permutation(len(y))
        ix=torch.as_tensor(order[(step%batches)*batch:((step%batches)+1)*batch],device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        schedule.before_update(step)
        logits,geometry,_=model.forward_features(tuple(x[ix] for x in features))
        loss=candidate['classification_weight']*balanced_loss(logits,labs[ix],positive) if use3 else 0
        if use6:
            loss=loss+(geometry[:,:,1:]-target[ix,:,1:]).square().sum(-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite joint training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),training['gradient_clip'])
        optimizer.step()
        schedule.after_update()
        if (step+1)%training['probe_every']==0 or step+1==training['updates']:
            record(step+1)
    assert best_state is not None
    model.load_state_dict(best_state)
    val,prob=evaluate_cached(model,vfeatures,vy,vlabels,positive)
    threshold=choose_threshold(vlabels,prob) if use3 else .5
    val,_=evaluate_cached(model,vfeatures,vy,vlabels,positive,threshold)
    train,_=evaluate_cached(model,features,y,labels,positive,threshold)
    diagnostics={}
    if model.arm!='geometry_only':
        for force in ('a','b'):
            diagnostics['forced_'+force]=evaluate_cached(model,vfeatures,vy,vlabels,positive,threshold,force)[0]
        diagnostics['shuffled_a']=evaluate_cached(model,(vfeatures[0].roll(1,0),vfeatures[1]),vy,vlabels,positive,threshold)[0]
    fit=dict(variant=variant,candidate=candidate,training=training,seed=seed,train_samples=len(y),
        validation_samples=len(vy),train_positive_fraction=positive,selected_step=best_step,
        threshold=threshold,train=train,validation=val,diagnostics=diagnostics,curve=history,
        parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-start,
        selection='minimum positive-step validation (1-AP) + RMSE/r for active tasks',
        objective='classification_weight * balanced BCE/log(2) + physical-radius MSE',
        original_train_variance=float(model.output_scale)**2,
        model_files='none; best state RAM only',joint_pair_dimensions=2*model.latent_dim,
        downstream_token_dimension=model.latent_dim)
    write_json(folder/'fit.json',fit)
    return model,fit
