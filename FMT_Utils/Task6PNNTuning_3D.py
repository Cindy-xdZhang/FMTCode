"""Version 1.2: larger PNN temporal networks and train-only regularization.

The Point-NN snapshot operator and the 1.1 baselines remain unchanged.
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from FMT_Utils.FlowMapData_3D import write_json
from FMT_Utils.Task6PNNTrans_3D import (PNNTransAutoencoder, fit_statistics,
    cached_features, score, predict)
from FMT_Utils.Task6DirectNeural_3D import check_geometry


def condition_features(features, mode):
    """Only training snapshots contribute; floor prevents tiny-channel whitening."""
    channels = features.shape[-1]
    if mode == 'none':
        return torch.zeros(channels, device=features.device), torch.ones(channels, device=features.device)
    if mode != 'floored_standardization':
        raise ValueError(mode)
    mean = features.double().mean((0, 1)).float()
    std = features.double().std((0, 1), unbiased=False).float()
    nonzero = std[std > 1e-6]
    floor = max(float(nonzero.median())*.05, 1e-3) if len(nonzero) else 1.
    return mean, std.clamp_min(floor)


class TunedPNNAutoencoder(PNNTransAutoencoder):
    def __init__(self, statistics, arm, candidate):
        super().__init__(statistics, arm=arm, dropout=candidate['dropout'], **candidate['architecture'])
        self.register_buffer('feature_mean', torch.zeros(self.input_projection.in_features))
        self.register_buffer('feature_scale', torch.ones(self.input_projection.in_features))
        width = candidate['architecture']['width']
        self.input_refinement = (nn.Sequential(nn.GELU(), nn.Dropout(candidate['dropout']), nn.Linear(width, width))
                                 if candidate['input_mlp'] else nn.Identity())
        self.input_drop = nn.Dropout(candidate['input_dropout'])
        self.feature_noise = candidate['feature_noise']

    def encode_features(self, features):
        q = (features-self.feature_mean)/self.feature_scale
        if self.training:
            if self.feature_noise:
                q = q + self.feature_noise*torch.randn_like(q)
            q = self.input_drop(q)
        x = self.input_refinement(self.input_projection(q)) + self.time_embedding
        for layer in self.temporal_layers:
            x = layer(x)
        return self.compress(self.latent_dropout(x.flatten(1)))


def mixed_geometry(geometry, fraction):
    """Convex combinations of training geometries only; targets stay clean.

    Synthetic trajectories are geometric augmentation, not new flow integrations.
    """
    if not 0 <= fraction <= 1:
        raise ValueError(fraction)
    count = int(len(geometry)*fraction)
    if not count:
        return geometry, count
    result = geometry.clone()
    partner = torch.randperm(len(geometry), device=geometry.device)[:count]
    weight = .2 + .6*torch.rand((count,1,1,1), device=geometry.device)
    result[:count] = weight*geometry[:count] + (1-weight)*geometry[partner]
    return result, count


def audit_tuned(model):
    assert not list(model.frontend.parameters())
    assert all(p.requires_grad for p in model.parameters())
    expected = {'feature_mean','feature_scale','output_mean','output_scale','initial_seeds',
                'frontend.position_mean','frontend.position_scale',
                'frontend.initial_frequencies','frontend.local_frequencies'}
    assert set(dict(model.named_buffers())) == expected
    return dict(arm=model.arm, frontend_parameters=0, snapshot_operator='frozen pnn_trans 1.1',
        trainable_parameters=sum(p.numel() for p in model.parameters()), latent_dimension=model.latent_dim,
        transformer_layers=len(model.temporal_layers), transformer_width=model.input_projection.out_features,
        decoder_input='latent only', analytic_inverse=False, pca_dependency=False, raw_skip=False,
        feature_scale_min=float(model.feature_scale.min()), feature_scale_max=float(model.feature_scale.max()))


def train_tuned(y, vy, arm, candidate, training, seed, device, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    check_geometry(y)
    check_geometry(vy)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    statistics = fit_statistics(y)
    model = TunedPNNAutoencoder(statistics, arm, candidate).to(device)
    x = cached_features(model, y, device)
    mean, scale = condition_features(x, candidate['feature_conditioning'])
    model.feature_mean.copy_(mean)
    model.feature_scale.copy_(scale)
    # Validation never contributes to feature conditioning or output statistics.
    vx = cached_features(model, vy, device)
    target = torch.as_tensor(y, device=device)
    structure = audit_tuned(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate['learning_rate'], weight_decay=candidate['weight_decay'])
    batch = min(training['batch_size'], len(y))
    batches = math.ceil(len(y)/batch)
    rng = np.random.default_rng(seed)
    start = time.monotonic()
    best, best_step, best_state = float('inf'), None, None
    curve = []

    def record(step, loss=None):
        nonlocal best, best_step, best_state
        fit, val = score(model,x,y,batch=128), score(model,vx,vy,batch=128)
        if not np.isfinite([fit,val]).all():
            raise FloatingPointError('Nonfinite geometry error')
        row = dict(step=step, train_rmse_r=fit, validation_rmse_r=val,
                   loss_normalized=loss, seconds=time.monotonic()-start)
        curve.append(row)
        if step > 0 and val < best:
            best, best_step = val, step
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        write_json(folder/'progress.json',row)
        print(dict(arm=arm,candidate=candidate['id'],**row),flush=True)

    record(0)
    exposed, augmented = 0, 0
    for step in range(training['updates']):
        if step % batches == 0:
            order = rng.permutation(len(y))
        ix = torch.as_tensor(order[(step%batches)*batch:((step%batches)+1)*batch],device=device)
        exposed += len(ix)
        warmup = min(1., (step+1)/max(1,training['warmup_updates']))
        rate = candidate['learning_rate']*warmup*(.1+.45*(1+math.cos(math.pi*step/max(1,training['updates']-1))))
        for group in optimizer.param_groups:
            group['lr'] = rate
        model.train()
        optimizer.zero_grad(set_to_none=True)
        truth, mixed_count = mixed_geometry(target[ix],candidate['geometry_mix_fraction'])
        augmented += mixed_count
        if mixed_count:
            # Re-encode the actual mixed geometry; mixing sine features is not equivalent.
            with torch.no_grad():
                features = model.snapshot_features(truth)
        else:
            features = x[ix]
        prediction = model.decode(model.encode_features(features))
        loss = (prediction[:,:,1:]-truth[:,:,1:]).square().sum(-1).mean()/model.output_scale.square()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),training['gradient_clip'])
        optimizer.step()
        if (step+1)%training['probe_every']==0 or step+1==training['updates']:
            record(step+1,loss.detach().item())
    assert best_state is not None
    model.load_state_dict(best_state)
    result = dict(arm=arm,candidate=candidate,seed=seed,structure=structure,train_samples=len(y),
        validation_samples=len(vy),statistics_train_samples=len(y),updates=training['updates'],
        train_examples_exposed=exposed,synthetic_geometries_exposed=augmented,selected_step=best_step,
        validation_rmse_r=best,train_rmse_r=score(model,x,y,batch=128),
        train_zero_latent_rmse_r=score(model,x,y,batch=128,zero_latent=True),curve=curve,
        seconds=time.monotonic()-start,model_files='none; best state RAM only',
        target_scale=statistics['output_scale'],position_scale=statistics['position_scale'],
        selection='lowest complete validation RMSE at positive steps; no test supplied')
    write_json(folder/'fit.json',result)
    return model,result
