"""Version 1.3 learning-rate policies; frozen 1.1/1.2 networks are reused."""
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from FMT_Utils.FlowMapData_3D import write_json
from FMT_Utils.Task6PNNTuning_3D import (TunedPNNAutoencoder, condition_features,
    mixed_geometry, audit_tuned, fit_statistics, cached_features, score, predict, check_geometry)


class LearningRateSchedule:
    """Apply batch schedules after optimizer.step; plateau observes validation only."""

    def __init__(self, optimizer, candidate, training):
        self.optimizer = optimizer
        self.spec = candidate['scheduler']
        self.kind = self.spec['name']
        self.base = candidate['learning_rate']
        self.updates = training['updates']
        self.warmup = training['warmup_updates']
        self.completed = 0
        self.last_used = None
        self.validation_steps = []
        self.scheduler = None
        if self.kind == 'one_cycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.base, total_steps=self.updates,
                pct_start=self.spec['pct_start'], anneal_strategy='cos',
                div_factor=self.spec['div_factor'], final_div_factor=self.spec['final_div_factor'],
                cycle_momentum=False)
        elif self.kind == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=self.spec['factor'], patience=self.spec['patience'],
                threshold=self.spec['threshold'], threshold_mode='rel',
                min_lr=self.base*self.spec['min_lr_ratio'])
        elif self.kind != 'cosine_legacy':
            raise ValueError(self.kind)

    @property
    def rate(self):
        return self.optimizer.param_groups[0]['lr']

    def before_update(self, step):
        assert step == self.completed and 0 <= step < self.updates
        rate = self.rate
        if self.kind == 'cosine_legacy':
            # Exactly the frozen 1.2 formula at the same total/warmup updates.
            warmup = min(1., (step+1)/max(1, self.warmup))
            rate = self.base*warmup*(.1+.45*(1+math.cos(math.pi*step/max(1, self.updates-1))))
        elif self.kind == 'plateau' and step < self.warmup:
            rate = self.base*(step+1)/max(1, self.warmup)
        for group in self.optimizer.param_groups:
            group['lr'] = rate
        self.last_used = rate
        return rate

    def after_update(self):
        self.completed += 1
        # Last optimization update uses the final scheduled rate. Do not create
        # an unused extra OneCycle step that rebounds after its minimum.
        if self.kind == 'one_cycle' and self.completed < self.updates:
            self.scheduler.step()

    def observe_validation(self, step, error):
        assert step == self.completed
        if self.kind == 'plateau' and step > 0 and step >= self.warmup:
            if self.validation_steps and step <= self.validation_steps[-1]:
                raise ValueError('A validation probe may be observed only once')
            self.scheduler.step(error)
            self.validation_steps.append(step)

    def record(self):
        state = self.scheduler.state_dict() if self.scheduler is not None else None
        if state is not None:
            # ReduceLROnPlateau stores +inf as its internal worst-value sentinel.
            # Encode this configuration sentinel as text, never as invalid JSON.
            state = {k: ('Infinity' if v > 0 else '-Infinity')
                     if isinstance(v, float) and math.isinf(v) else v for k,v in state.items()}
        return dict(spec=self.spec, base_or_max_learning_rate=self.base, completed_updates=self.completed,
                    last_learning_rate_used=self.last_used, validation_steps=self.validation_steps,
                    scheduler_state=state,
                    torch_version=torch.__version__, cycle_momentum=False)


def train_scheduled(y, vy, arm, candidate, training, seed, device, folder):
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
    schedule = LearningRateSchedule(optimizer, candidate, training)
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
        schedule.observe_validation(step, val)
        row = dict(step=step, learning_rate_used=schedule.last_used, learning_rate_after_probe=schedule.rate,
                   train_rmse_r=fit, validation_rmse_r=val,
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
        schedule.before_update(step)
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
        schedule.after_update()
        if (step+1)%training['probe_every']==0 or step+1==training['updates']:
            record(step+1,loss.detach().item())
    assert best_state is not None
    model.load_state_dict(best_state)
    result = dict(arm=arm,candidate=candidate,seed=seed,structure=structure,train_samples=len(y),
        validation_samples=len(vy),statistics_train_samples=len(y),updates=training['updates'],
        train_examples_exposed=exposed,synthetic_geometries_exposed=augmented,selected_step=best_step,
        validation_rmse_r=best,train_rmse_r=score(model,x,y,batch=128),
        train_zero_latent_rmse_r=score(model,x,y,batch=128,zero_latent=True),curve=curve,
        schedule=schedule.record(), training=training,
        seconds=time.monotonic()-start,model_files='none; best state RAM only',
        target_scale=statistics['output_scale'],position_scale=statistics['position_scale'],
        selection='lowest complete validation RMSE at positive steps; no test supplied')
    write_json(folder/'fit.json',result)
    return model,result
