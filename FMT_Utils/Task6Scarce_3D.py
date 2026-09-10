"""Small-data Task6 helpers; the 3.1 encoder and VAE implementation stay frozen."""
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from FMT_Utils.Task6Recovery_3D import initialize_vae, signed_fmt
from FMT_Utils.PrimitiveVAE_3D import vae_loss
from FMT_Utils.FlowMapData_3D import write_json


def subset_order(count, seed):
    """A geometry-independent nested random subset, shared by both arms."""
    return np.random.default_rng(seed).permutation(count)


def inputs_numpy(geometry, arm, frequencies):
    if arm == 'raw_vae':
        return geometry.reshape(len(geometry), 672).copy()
    if arm != 'signed_fmt_vae':
        raise ValueError(arm)
    return signed_fmt(geometry, frequencies).astype(np.float32)


def inputs_torch(geometry, arm, frequencies):
    if arm == 'raw_vae':
        return geometry.flatten(1)
    if arm != 'signed_fmt_vae':
        raise ValueError(arm)
    x = geometry.to(torch.float64)
    relative = x[:, 1:] - x[:, :1]
    signal = torch.cat([x[:, :1].diff(dim=2), relative.diff(dim=2)], dim=1)
    spectrum = torch.fft.rfft(signal, dim=2, norm='forward')[:, :, :frequencies]
    return torch.cat([spectrum.real.flatten(1), spectrum.imag[:, :, 1:].flatten(1)], dim=1).float()


def add_dropout(module, probability):
    """Drop hidden activations in both residual networks; leave frozen maps alone."""
    if not 0 <= probability < 1:
        raise ValueError('Dropout probability must be in [0,1)')
    for name, child in list(module.named_children()):
        add_dropout(child, probability)
        if isinstance(child, nn.SiLU) and probability:
            setattr(module, name, nn.Sequential(child, nn.Dropout(probability)))


def perturb_geometry(y, sigma, generator):
    """Denoising input only; initial protocol seeds and clean targets are preserved."""
    if sigma == 0:
        return y
    noise = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=generator) * sigma
    noise[:, :, 0] = 0
    return y + noise


@torch.no_grad()
def primary_score(model, x, y, device, batch=1024):
    model.eval()
    total = 0.
    count = 0
    for begin in range(0, len(x), batch):
        output, _, _ = model(torch.as_tensor(x[begin:begin+batch], device=device), sample=False)
        target = torch.as_tensor(y[begin:begin+batch], device=device)
        diff = output[:, :, 1:].double() - target[:, :, 1:].double()
        total += diff.square().sum().item()
        count += len(diff) * 7 * 31
    return math.sqrt(total / count)


def fit_small(y, vy, arm, candidate, training, seed, device, folder):
    """Only training/validation arrays enter fitting; models never go to disk."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    frequencies = candidate['frequencies']
    x = inputs_numpy(y, arm, frequencies)
    vx = inputs_numpy(vy, arm, frequencies)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model, initialization = initialize_vae(x, y, arm, training['latent_dim'], frequencies,
        training['width'], training['blocks'])
    add_dropout(model.encoder_residual, candidate['dropout'])
    add_dropout(model.decoder_residual, candidate['dropout'])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
        weight_decay=candidate['weight_decay'])
    xt = torch.as_tensor(x, device=device)
    yt = torch.as_tensor(y, device=device)
    rng = np.random.default_rng(seed)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 300000)
    batch = min(training['batch_size'], len(y))
    per_epoch = math.ceil(len(y) / batch)
    best_state = None
    best_value = float('inf')
    best_step = None
    curve = []
    start = time.monotonic()

    def record(step, reconstruction=None, kl=None):
        nonlocal best_state, best_value, best_step
        validation = primary_score(model, vx, vy, device)
        train_score = primary_score(model, x, y, device)
        row = dict(step=step, train_rmse_r=train_score, validation_rmse_r=validation,
            reconstruction=reconstruction, kl=kl, seconds=time.monotonic()-start)
        if not np.isfinite(validation):
            raise FloatingPointError('Nonfinite validation error')
        curve.append(row)
        if step > 0 and validation < best_value:
            best_value, best_step = validation, step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        write_json(folder/'progress.json', row)
        print(dict(arm=arm, candidate=candidate['id'], **row), flush=True)

    record(0)
    for step in range(training['updates']):
        if step % per_epoch == 0:
            order = rng.permutation(len(y))
        index = torch.as_tensor(order[(step % per_epoch)*batch:((step % per_epoch)+1)*batch], device=device)
        rate = training['learning_rate'] * (.1 + .45*(1 + math.cos(math.pi*step/max(1, training['updates']-1))))
        for group in optimizer.param_groups:
            group['lr'] = rate
        model.train()
        optimizer.zero_grad(set_to_none=True)
        clean = yt[index]
        inputs = inputs_torch(perturb_geometry(clean, candidate['noise_sigma'], noise_generator), arm, frequencies) if candidate['noise_sigma'] else xt[index]
        output, mean, logvar = model(inputs, sample=True)
        loss, rec, kl = vae_loss(output, clean, mean, logvar, training['beta'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'])
        optimizer.step()
        if (step+1) % training['probe_every'] == 0 or step+1 == training['updates']:
            record(step+1, rec.detach().item(), kl.detach().item())
    assert best_state is not None and best_step > 0
    model.load_state_dict(best_state)
    model.eval()
    report = dict(candidate=candidate, arm=arm, seed=seed, train_samples=len(y),
        validation_samples=len(vy), initialization=initialization, updates=training['updates'],
        train_examples_exposed=(training['updates']//per_epoch)*len(y)+min(len(y),(training['updates']%per_epoch)*batch),
        selected_step=best_step, validation_rmse_r=best_value, curve=curve,
        train_rmse_r=primary_score(model,x,y,device), dropout_eval_disabled=True,
        selection='minimum validation RMSE at positive training steps; no test arrays in fitting',
        model_files='none', seconds=time.monotonic()-start)
    write_json(folder/'fit.json', report)
    return model, report
