"""FMT/geometry latent mixture of experts, version 1.1; no analytic inverse."""
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.FlowMapData_3D import cross_offsets, write_json
from FMT_Utils.Task6DirectNeural_3D import check_geometry, LearnedResidualBlock
from FMT_Utils.Task6PNNTrans_3D import SnapshotPointNN, fit_statistics
from FMT_Utils.Task6PNNSchedule_3D import LearningRateSchedule


def original_fmt(geometry):
    """Exactly the old 161D fmt_all recipe, not signed/full Fourier coefficients."""
    return pathline_dft_features_3d(geometry, num_freq=6, neighbor_weight=.5,
        neighbor_scale=100., neighbor_pool='sort', mode='gram',
        include_chirality=True, return_numpy=False)


def channel_statistics(x):
    dims = tuple(range(x.ndim-1))
    mean = x.double().mean(dims).float()
    std = x.double().std(dims, unbiased=False).float()
    active = std[std>1e-6]
    floor = max(float(active.median())*.05, .001) if len(active) else 1.
    return mean, std.clamp_min(floor)


class DenseEncoder(nn.Module):
    def __init__(self, inputs, latent, width, blocks, dropout):
        super().__init__()
        self.linear = nn.Linear(inputs, latent)
        self.nonlinear = nn.Sequential(nn.Linear(inputs, width),
            *[LearnedResidualBlock(width, dropout) for _ in range(blocks)],
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, latent))

    def forward(self, x):
        return self.linear(x) + .1*self.nonlinear(x)


class TemporalEncoder(nn.Module):
    def __init__(self, inputs, latent, spec, dropout):
        super().__init__()
        width = spec['temporal_width']
        self.projection = nn.Linear(inputs, width)
        self.position = nn.Parameter(torch.randn(1,32,width)*.02)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(width,spec['temporal_heads'],
            4*width,dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
            for _ in range(spec['temporal_layers'])])
        self.compress = nn.Sequential(nn.Dropout(dropout),nn.Linear(32*width,latent))

    def forward(self, x):
        x = self.projection(x)+self.position
        for layer in self.layers:
            x = layer(x)
        return self.compress(x.flatten(1))


class FMTGeometryMoE(nn.Module):
    """Two learned latent experts, input-dependent routing and a shared decoder."""
    def __init__(self, statistics, candidate, arm):
        super().__init__()
        if arm not in ('fmt_moe','geometry_moe','geometry_only'):
            raise ValueError(arm)
        if candidate['gate'] not in ('scalar','channel'):
            raise ValueError(candidate['gate'])
        self.arm, self.candidate = arm, candidate
        spec, drop = candidate['architecture'], candidate['dropout']
        self.latent_dim = spec['latent_dim']
        self.family = candidate['geometry_branch']
        self.frontend = SnapshotPointNN(statistics['position_mean'],statistics['position_scale'])
        self.register_buffer('output_mean',torch.as_tensor(statistics['output_mean'],dtype=torch.float32))
        self.register_buffer('output_scale',torch.as_tensor(statistics['output_scale'],dtype=torch.float32))
        self.register_buffer('initial_seeds',torch.as_tensor(cross_offsets(),dtype=torch.float32))
        bdim = 651 if self.family == 'raw_mlp' else self.frontend.output_dim
        for name,dim in [('a',651),('b',bdim)]:
            self.register_buffer(name+'_mean',torch.zeros(dim))
            self.register_buffer(name+'_scale',torch.ones(dim))
        # Construct B and decoder first: their initialization is identical in all arms.
        if self.family == 'raw_mlp':
            self.geometry = DenseEncoder(bdim,self.latent_dim,spec['expert_width'],spec['expert_blocks'],drop)
        elif self.family == 'pointnn':
            self.geometry = TemporalEncoder(bdim,self.latent_dim,spec,drop)
        else:
            raise ValueError(self.family)
        self.output_linear = nn.Linear(self.latent_dim,651)
        self.output_network = nn.Sequential(nn.Linear(self.latent_dim,spec['decoder_width']),
            *[LearnedResidualBlock(spec['decoder_width'],drop) for _ in range(spec['decoder_blocks'])],
            nn.SiLU(),nn.Dropout(drop),nn.Linear(spec['decoder_width'],651))
        for layer in (self.output_linear,self.output_network[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.gate_channels = 1 if candidate['gate']=='scalar' else self.latent_dim
        if arm != 'geometry_only':
            self.expert_a = DenseEncoder(651,self.latent_dim,spec['expert_width'],spec['expert_blocks'],drop)
            self.router = nn.Sequential(nn.LayerNorm(2*self.latent_dim),
                nn.Linear(2*self.latent_dim,spec['gate_width']),nn.GELU(),
                nn.Linear(spec['gate_width'],2*self.gate_channels))
            nn.init.zeros_(self.router[-1].weight)
            with torch.no_grad():
                self.router[-1].bias.copy_(torch.tensor([math.log(.1),math.log(.9)]).repeat(self.gate_channels))

    @torch.no_grad()
    def features(self, y):
        raw = y[:,:,1:].flatten(1)
        # Zero padding gives exactly equal A-layer capacity in FMT and Raw controls.
        a = F.pad(original_fmt(y),(0,651-161)) if self.arm=='fmt_moe' else raw
        b = raw if self.family=='raw_mlp' else self.frontend(y.transpose(1,2))
        return a,b

    def condition(self, features):
        for name,x in zip(('a','b'),features):
            mean,scale = channel_statistics(x)
            getattr(self,name+'_mean').copy_(mean)
            getattr(self,name+'_scale').copy_(scale)

    def encode_features(self, features, route=None):
        a,b = features
        zb = self.geometry((b-self.b_mean)/self.b_scale)
        if self.arm=='geometry_only':
            return zb,dict(a=None,b=zb,weight_a=torch.zeros(len(zb),1,device=zb.device))
        za = self.expert_a((a-self.a_mean)/self.a_scale)
        gates = self.router(torch.cat((za,zb),-1)).reshape(-1,self.gate_channels,2).softmax(-1)
        ga = gates[...,0]
        if route=='geometry':
            ga = torch.zeros_like(ga)
        elif route=='expert_a':
            ga = torch.ones_like(ga)
        elif route is not None:
            raise ValueError(route)
        return ga*za+(1-ga)*zb,dict(a=za,b=zb,weight_a=ga)

    def decode(self,z):
        later = ((self.output_linear(z)+.1*self.output_network(z))*self.output_scale+self.output_mean).reshape(-1,7,31,3)
        seeds = self.initial_seeds[None,:,None].expand(len(z),-1,-1,-1)
        return torch.cat((seeds,later),2)

    def forward(self,y):
        z,_ = self.encode_features(self.features(y))
        return self.decode(z)


@torch.no_grad()
def cached(model,y,device,batch=128):
    parts = [[],[]]
    for begin in range(0,len(y),batch):
        values = model.features(torch.as_tensor(y[begin:begin+batch],device=device))
        for out,x in zip(parts,values):
            out.append(x)
    return tuple(torch.cat(p) for p in parts)


@torch.no_grad()
def score(model,x,y,route=None,zero_latent=False,batch=128):
    model.eval()
    total,count,weights = 0.,0,[]
    for begin in range(0,len(y),batch):
        z,aux = model.encode_features(tuple(p[begin:begin+batch] for p in x),route)
        out = model.decode(torch.zeros_like(z) if zero_latent else z)
        target = torch.as_tensor(y[begin:begin+batch],device=out.device)
        total += (out[:,:,1:].double()-target[:,:,1:].double()).square().sum().item()
        count += len(out)*7*31
        weights.append(aux['weight_a'].detach().cpu())
    w = torch.cat(weights).double()
    return math.sqrt(total/count),dict(a_mean=float(w.mean()),a_std=float(w.std(unbiased=False)),
        fraction_a_below_001=float((w<.01).double().mean()),fraction_a_above_099=float((w>.99).double().mean()))


@torch.no_grad()
def predict(model,y,device,batch=128):
    model.eval()
    return np.concatenate([model(torch.as_tensor(y[i:i+batch],device=device)).cpu().numpy()
                           for i in range(0,len(y),batch)])


def audit_model(model):
    assert not list(model.frontend.parameters())
    assert all(p.requires_grad for p in model.parameters())
    return dict(arm=model.arm,geometry_branch=model.family,latent_dimension=model.latent_dim,
        trainable_parameters=sum(p.numel() for p in model.parameters()),fixed_pointnn_parameters=0,
        original_fmt_dimension=161,expert_a_input_dimension=651,gate=model.candidate['gate'],
        decoder_input='mixed latent only',analytic_inverse=False,pca=False,raw_geometry_skip=False)


def train_moe(y,vy,arm,candidate,training,seed,device,folder):
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    check_geometry(y)
    check_geometry(vy)
    torch.manual_seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(seed)
    model = FMTGeometryMoE(fit_statistics(y),candidate,arm).to(device)
    x = cached(model,y,device)
    model.condition(x)
    vx = cached(model,vy,device)
    target = torch.as_tensor(y,device=device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=candidate['learning_rate'],weight_decay=candidate['weight_decay'])
    schedule = LearningRateSchedule(optimizer,candidate,training)
    batch = min(len(y),training['batch_size'])
    batches = math.ceil(len(y)/batch)
    rng = np.random.default_rng(seed)
    start = time.monotonic()
    best,best_step,best_state = float('inf'),None,None
    curve = []

    def record(step,loss=None):
        nonlocal best,best_step,best_state
        fit,_ = score(model,x,y)
        val,gates = score(model,vx,vy)
        if not np.isfinite([fit,val]).all():
            raise FloatingPointError('Nonfinite geometry RMSE')
        schedule.observe_validation(step,val)
        row = dict(step=step,train_rmse_r=fit,validation_rmse_r=val,gate=gates,
                   learning_rate_used=schedule.last_used,loss=loss,seconds=time.monotonic()-start)
        curve.append(row)
        if step>0 and val<best:
            best,best_step = val,step
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        write_json(folder/'progress.json',row)
        print(dict(arm=arm,candidate=candidate['id'],**row),flush=True)

    record(0)
    exposed = 0
    for step in range(training['updates']):
        if step%batches==0:
            order = rng.permutation(len(y))
        ix = torch.as_tensor(order[(step%batches)*batch:((step%batches)+1)*batch],device=device)
        exposed += len(ix)
        schedule.before_update(step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z,aux = model.encode_features(tuple(p[ix] for p in x))
        truth = target[ix,:,1:]
        def error(latent):
            return (model.decode(latent)[:,:,1:]-truth).square().sum(-1).mean()/model.output_scale.square()
        loss = error(z)
        # Early shared-decoder supervision gives both experts useful gradients.
        # It vanishes after the first quarter; no uniform-routing constraint.
        weight = candidate['auxiliary_weight']*max(0.,1-(step+1)/max(1,.25*training['updates']))
        if weight and aux['a'] is not None:
            loss = loss+weight*.5*(error(aux['a'])+error(aux['b']))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),training['gradient_clip'])
        optimizer.step()
        schedule.after_update()
        if (step+1)%training['probe_every']==0 or step+1==training['updates']:
            record(step+1,float(loss.detach()))
    assert best_state is not None
    model.load_state_dict(best_state)
    train_error,_ = score(model,x,y)
    val_error,gates = score(model,vx,vy)
    diagnostic = {}
    if arm!='geometry_only':
        for route in ('geometry','expert_a'):
            diagnostic[route+'_only_rmse_r'] = score(model,vx,vy,route=route)[0]
        diagnostic['shuffled_a_rmse_r'] = score(model,(vx[0].roll(1,0),vx[1]),vy)[0]
    result = dict(arm=arm,candidate=candidate,training=training,seed=seed,structure=audit_model(model),
        train_samples=len(y),validation_samples=len(vy),statistics_train_samples=len(y),
        updates=training['updates'],train_examples_exposed=exposed,selected_step=best_step,
        train_rmse_r=train_error,validation_rmse_r=val_error,validation_gates=gates,
        validation_branch_diagnostics=diagnostic,train_zero_latent_rmse_r=score(model,x,y,zero_latent=True)[0],
        schedule=schedule.record(),curve=curve,seconds=time.monotonic()-start,
        model_files='none; best state RAM only',selection='lowest validation RMSE at positive steps')
    write_json(folder/'fit.json',result)
    return model,result
