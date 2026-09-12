"""Task3 classification and Task6 reconstruction with shared experts and two gates."""
import copy
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,f1_score

from FMT_Utils.Task6FMTGeometryMoE_3D import FMTGeometryMoE,cached,fit_statistics
from FMT_Utils.Task6DirectNeural_3D import check_geometry,LearnedResidualBlock
from FMT_Utils.Task6PNNSchedule_3D import LearningRateSchedule
from FMT_Utils.FlowMapData_3D import write_json

VARIANTS=('joint_task_gates','joint_shared_gate','joint_raw_task_gates',
          'joint_fixed_routes','joint_geometry_only','single_task3','single_task6')


class Task36MultiGate(FMTGeometryMoE):
    def __init__(self,statistics,candidate,variant):
        assert variant in VARIANTS
        assert candidate['geometry_branch']=='raw_mlp' and candidate['gate']=='scalar'
        base='geometry_moe' if variant=='joint_raw_task_gates' else 'geometry_only' if variant=='joint_geometry_only' else 'fmt_moe'
        super().__init__(statistics,candidate,base)
        self.variant=variant
        # A separate gate identifies the task without reading either target.
        if base!='geometry_only':
            with torch.no_grad():
                self.router[-1].bias.zero_()
            self.classification_router=copy.deepcopy(self.router)
        width=candidate['classification_width']
        self.classifier=nn.Sequential(nn.LayerNorm(self.latent_dim),nn.Linear(self.latent_dim,width),
            LearnedResidualBlock(width,candidate['dropout']),nn.SiLU(),nn.Linear(width,1))

    def encode_tasks(self,features,force=None):
        a,b=features
        zb=self.geometry((b-self.b_mean)/self.b_scale)
        if self.arm=='geometry_only':
            zero=torch.zeros(len(zb),1,device=zb.device)
            return zb,zb,dict(task3=zero,task6=zero)
        za=self.expert_a((a-self.a_mean)/self.a_scale)
        inputs=torch.cat((za,zb),-1)
        g6=self.router(inputs).softmax(-1)[:,:1]
        g3=(self.router if self.variant=='joint_shared_gate' else self.classification_router)(inputs).softmax(-1)[:,:1]
        if self.variant=='joint_fixed_routes':
            g3,g6=torch.ones_like(g3),torch.zeros_like(g6)
        if force is not None:
            if force=='a':
                g3,g6=torch.ones_like(g3),torch.ones_like(g6)
            elif force=='b':
                g3,g6=torch.zeros_like(g3),torch.zeros_like(g6)
            else:
                raise ValueError(force)
        return g3*za+(1-g3)*zb,g6*za+(1-g6)*zb,dict(task3=g3,task6=g6)

    def forward_features(self,features,force=None):
        z3,z6,gates=self.encode_tasks(features,force)
        return self.classifier(z3).flatten(),self.decode(z6),gates

    def forward(self,y):
        return self.forward_features(self.features(y))


def task_flags(variant):
    return variant!='single_task6',variant!='single_task3'


def classification_metrics(labels,probability,threshold=.5):
    labels=np.asarray(labels,dtype=np.int64)
    probability=np.asarray(probability,dtype=np.float64)
    assert set(np.unique(labels))<={0,1} and np.isfinite(probability).all()
    both=len(np.unique(labels))==2
    return dict(average_precision=float(average_precision_score(labels,probability)) if both else None,
        auroc=float(roc_auc_score(labels,probability)) if both else None,f1=float(f1_score(labels,probability>=threshold,zero_division=0)),
        positive_fraction=float(labels.mean()),threshold=float(threshold))


def choose_threshold(labels,probability):
    precision,recall,thresholds=precision_recall_curve(labels,probability)
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    return float(thresholds[np.flatnonzero(f1>=f1.max()-1e-12)[-1]])


def balanced_loss(logits,labels,positive_fraction):
    weights=torch.where(labels>0,1/(2*positive_fraction),1/(2*(1-positive_fraction)))
    return (F.binary_cross_entropy_with_logits(logits,labels,reduction='none')*weights).mean()/math.log(2)


@torch.no_grad()
def evaluate_cached(model,features,geometry,labels,positive_fraction,threshold=.5,force=None,batch=256):
    model.eval()
    use3,use6=task_flags(model.variant)
    total=0.
    probability,weights3,weights6=[],[],[]
    cls_sum=0.
    for begin in range(0,len(geometry),batch):
        inputs=tuple(x[begin:begin+batch] for x in features)
        logits,pred,gates=model.forward_features(inputs,force)
        n=len(pred)
        target=torch.as_tensor(geometry[begin:begin+batch],device=pred.device)
        total+=(pred[:,:,1:].double()-target[:,:,1:].double()).square().sum().item()
        lab=torch.as_tensor(labels[begin:begin+batch],device=pred.device,dtype=torch.float32)
        cls_sum+=balanced_loss(logits,lab,positive_fraction).item()*n
        probability.append(logits.sigmoid().cpu().numpy())
        weights3.append(gates['task3'].cpu().numpy())
        weights6.append(gates['task6'].cpu().numpy())
    probability=np.concatenate(probability)
    rmse=math.sqrt(total/(len(geometry)*7*31))
    cls=cls_sum/len(geometry)
    metric=classification_metrics(labels,probability,threshold) if use3 else None
    score=(cls if use3 else 0)+(rmse**2/float(model.output_scale)**2 if use6 else 0)
    return dict(task3=metric,task6_rmse_r=rmse if use6 else None,selection_score=score,
        normalized_classification_loss=cls if use3 else None,
        gate3_mean=float(np.concatenate(weights3).mean()),gate6_mean=float(np.concatenate(weights6).mean())),probability


@torch.no_grad()
def predict(model,y,device,batch=256,force=None):
    model.eval()
    probabilities,predictions=[],[]
    for begin in range(0,len(y),batch):
        logits,geometry,_=model.forward_features(model.features(torch.as_tensor(y[begin:begin+batch],device=device)),force)
        probabilities.append(logits.sigmoid().cpu().numpy())
        predictions.append(geometry.cpu().numpy())
    return np.concatenate(probabilities),np.concatenate(predictions)


def train_joint(y,labels,vy,vlabels,variant,candidate,training,seed,device,folder):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    check_geometry(y)
    check_geometry(vy)
    assert len(labels)==len(y) and len(vlabels)==len(vy)
    assert set(np.unique(labels))=={0,1} and set(np.unique(vlabels))=={0,1}
    torch.manual_seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(seed)
    model=Task36MultiGate(fit_statistics(y),candidate,variant).to(device)
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
        loss=balanced_loss(logits,labs[ix],positive) if use3 else 0
        if use6:
            loss=loss+(geometry[:,:,1:]-target[ix,:,1:]).square().sum(-1).mean()/model.output_scale.square()
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
        selection='minimum positive-step validation sum of train-normalized active task losses',
        model_files='none; best state RAM only',joint_pair_dimensions=2*model.latent_dim,
        downstream_token_dimension=model.latent_dim)
    write_json(folder/'fit.json',fit)
    return model,fit
