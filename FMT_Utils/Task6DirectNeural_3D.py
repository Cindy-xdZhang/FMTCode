"""Task6 5.1: learned geometry compression with no analytic reconstruction path."""
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from FMT_Utils.FlowMapData_3D import cross_offsets, write_json


def check_geometry(geometry):
    x = np.asarray(geometry)
    if x.ndim != 4 or x.shape[1:] != (7, 32, 3) or not np.isfinite(x).all():
        raise ValueError('Expected finite [N,7,32,3] geometry')
    np.testing.assert_allclose(x[:, :, 0], np.broadcast_to(cross_offsets(), (len(x), 7, 3)), atol=2e-5)
    return x


def direct_inputs(geometry, arm):
    x = check_geometry(geometry)
    if arm == 'raw_neural':
        return x[:, :, 1:].reshape(len(x), 651).astype(np.float32)
    if arm != 'signed_fmt16_neural':
        raise ValueError(arm)
    x = x.astype(np.float64)
    relative = x[:, 1:] - x[:, :1]
    signal = np.concatenate([np.diff(x[:, :1], axis=2), np.diff(relative, axis=2)], axis=1)
    spectrum = np.fft.rfft(signal, axis=2, norm='forward')
    return np.concatenate([spectrum.real.reshape(len(x), -1),
        spectrum.imag[:, :, 1:].reshape(len(x), -1)], axis=1).astype(np.float32)


def fit_statistics(inputs, train_geometry):
    """Only per-channel offsets/scales and one output scale; no matrix fitting."""
    y = check_geometry(train_geometry)[:, :, 1:].astype(np.float64)
    mean = y.mean(0)
    scale = max(float(np.sqrt(np.mean(np.sum((y - mean) ** 2, axis=-1)))), 1e-3)
    input_std = inputs.std(0, dtype=np.float64)
    input_std[input_std < 1e-6] = 1.
    return dict(input_mean=inputs.mean(0, dtype=np.float64), input_std=input_std,
        output_mean=mean.reshape(651), output_scale=np.array(scale))


class LearnedResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, width),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, width))

    def forward(self, x):
        return x + .1 * self.layers(x)


class DirectGeometryVAE(nn.Module):
    """All input-dependent mappings are learned; decode accepts only the latent."""
    def __init__(self, statistics, latent_dim=192, width=512, blocks=3, dropout=.1):
        super().__init__()
        for name, value in statistics.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.register_buffer('initial_seeds', torch.as_tensor(cross_offsets(), dtype=torch.float32))
        self.encoder_projection = nn.Linear(651, latent_dim)
        self.encoder_network = nn.Sequential(nn.Linear(651, width),
            *[LearnedResidualBlock(width, dropout) for _ in range(blocks)],
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, latent_dim))
        self.decoder_projection = nn.Linear(latent_dim, 651)
        self.decoder_network = nn.Sequential(nn.Linear(latent_dim, width),
            *[LearnedResidualBlock(width, dropout) for _ in range(blocks)],
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, 651))
        self.posterior_logvar = nn.Parameter(torch.full((latent_dim,), -18.))
        # The untrained decoder predicts the train mean, independent of input.
        for layer in (self.decoder_projection, self.decoder_network[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def encode(self, x):
        q = (x - self.input_mean) / self.input_std
        mean = self.encoder_projection(q) + .1 * self.encoder_network(q)
        return mean, self.posterior_logvar.clamp(-24., 0.).expand_as(mean)

    def decode(self, z):
        q = self.decoder_projection(z) + .1 * self.decoder_network(z)
        later = (q * self.output_scale + self.output_mean).reshape(-1, 7, 31, 3)
        initial = self.initial_seeds[None, :, None].expand(len(z), -1, -1, -1)
        return torch.cat([initial, later], dim=2)

    def forward(self, x, sample=True):
        mean, logvar = self.encode(x)
        z = mean + torch.randn_like(mean) * torch.exp(.5 * logvar) if sample else mean
        return self.decode(z), mean, logvar


def audit_direct_model(model):
    allowed = {'input_mean': (651,), 'input_std': (651,), 'output_mean': (651,),
        'output_scale': (), 'initial_seeds': (7, 3)}
    buffers = dict(model.named_buffers())
    assert set(buffers) == set(allowed)
    assert all(tuple(buffers[k].shape) == shape for k, shape in allowed.items())
    assert all(p.requires_grad for p in model.parameters())
    return dict(trainable_parameters=sum(p.numel() for p in model.parameters()),
        fixed_buffers={k: list(v.shape) for k, v in buffers.items()},
        analytic_inverse_path=False, pca_in_neural_model=False,
        input_dimension=651, latent_dimension=model.encoder_projection.out_features)


@torch.no_grad()
def predict_direct(model, inputs, device, batch=1024, zero_latent=False):
    model.eval()
    result = []
    for begin in range(0, len(inputs), batch):
        mean, _ = model.encode(torch.as_tensor(inputs[begin:begin + batch], device=device))
        prediction = model.decode(torch.zeros_like(mean) if zero_latent else mean)
        result.append(prediction.cpu().numpy())
    return np.concatenate(result)


@torch.no_grad()
def direct_score(model, inputs, geometry, device, batch=1024, zero_latent=False):
    model.eval()
    total, count = 0., 0
    for begin in range(0, len(inputs), batch):
        x = torch.as_tensor(inputs[begin:begin + batch], device=device)
        y = torch.as_tensor(geometry[begin:begin + batch], device=device)
        mean, _ = model.encode(x)
        prediction = model.decode(torch.zeros_like(mean) if zero_latent else mean)
        delta = prediction[:, :, 1:].double() - y[:, :, 1:].double()
        total += delta.square().sum().item()
        count += len(delta) * 7 * 31
    return math.sqrt(total / count)


def train_direct(y, vy, arm, candidate, training, seed, device, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    x, vx = direct_inputs(y, arm), direct_inputs(vy, arm)
    statistics = fit_statistics(x, y)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = DirectGeometryVAE(statistics, training['latent_dim'], training['width'],
        training['blocks'], candidate['dropout']).to(device)
    structure = audit_direct_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate['learning_rate'],
        weight_decay=candidate['weight_decay'])
    xt, yt = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
    rng = np.random.default_rng(seed)
    batch = min(training['batch_size'], len(y))
    batches = math.ceil(len(y) / batch)
    best, best_step, best_state = float('inf'), None, None
    curve = []
    start = time.monotonic()

    def record(step, reconstruction=None, kl=None):
        nonlocal best, best_step, best_state
        value = direct_score(model, vx, vy, device)
        fitting = direct_score(model, x, y, device)
        if not np.isfinite([value, fitting]).all():
            raise FloatingPointError('Nonfinite geometry error')
        row = dict(step=step, validation_rmse_r=value, train_rmse_r=fitting,
            reconstruction_normalized=reconstruction, kl=kl, seconds=time.monotonic() - start)
        curve.append(row)
        if step > 0 and value < best:
            best, best_step = value, step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        write_json(folder / 'progress.json', row)
        print(dict(arm=arm, candidate=candidate['id'], **row), flush=True)

    record(0)
    for step in range(training['updates']):
        if step % batches == 0:
            order = rng.permutation(len(y))
        ix = torch.as_tensor(order[(step % batches)*batch:((step % batches)+1)*batch], device=device)
        rate = candidate['learning_rate'] * (.1 + .45 * (1 + math.cos(math.pi * step / max(1, training['updates'] - 1))))
        for group in optimizer.param_groups:
            group['lr'] = rate
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, mean, logvar = model(xt[ix], sample=True)
        reconstruction = (prediction[:, :, 1:] - yt[ix, :, 1:]).square().sum(-1).mean() / model.output_scale.square()
        kl = .5 * (mean.square() + logvar.exp() - 1. - logvar).mean()
        loss = reconstruction + training['beta'] * kl
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'])
        optimizer.step()
        if (step + 1) % training['probe_every'] == 0 or step + 1 == training['updates']:
            record(step + 1, reconstruction.detach().item(), kl.detach().item())
    assert best_state is not None and best_step > 0
    model.load_state_dict(best_state)
    model.eval()
    result = dict(arm=arm, candidate=candidate, seed=seed, structure=structure,
        train_samples=len(y), validation_samples=len(vy), statistics_train_samples=len(y),
        target_scale=float(statistics['output_scale']), updates=training['updates'],
        train_examples_exposed=(training['updates']//batches)*len(y)+min(len(y),(training['updates']%batches)*batch),
        selected_step=best_step, validation_rmse_r=best, train_rmse_r=direct_score(model, x, y, device),
        train_zero_latent_rmse_r=direct_score(model, x, y, device, zero_latent=True), curve=curve,
        model_files='none; best state in RAM', seconds=time.monotonic()-start,
        selection='lowest validation RMSE at positive steps; no test inputs supplied to fitting')
    write_json(folder / 'fit.json', result)
    return model, result
