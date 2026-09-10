"""Full frozen FMT input and explicit training-fit diagnostics, version 1.1."""
from __future__ import annotations

import math
import time
import numpy as np
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, x):
        return x + .1 * self.layers(x)


class DirectFMTDecoder(nn.Module):
    """All 161 channels enter the network; normalization is invertible and internal."""
    def __init__(self, mean, std, hidden=512, support_blocks=2, query_blocks=4, output_scale=1., small=False):
        super().__init__()
        self.register_buffer('input_mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('input_std', torch.as_tensor(std, dtype=torch.float32))
        self.output_scale = float(output_scale)
        dim = len(mean)
        if small:
            self.support = nn.Sequential(nn.Linear(dim + 4, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
            self.query = nn.Sequential(nn.Linear(hidden + 6, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        else:
            self.support = nn.Sequential(nn.Linear(dim + 4, hidden), *[ResidualBlock(hidden) for _ in range(support_blocks)], nn.SiLU())
            self.query = nn.Sequential(nn.Linear(hidden + 6, hidden), *[ResidualBlock(hidden) for _ in range(query_blocks)], nn.SiLU(), nn.Linear(hidden, 3))

    def encode_support(self, tokens, geometry):
        values = (tokens - self.input_mean) / self.input_std
        return self.support(torch.cat([values, geometry], -1)).mean(1)

    def query_context(self, context, query, tau, scales):
        value = self.query(torch.cat([context, query, tau[:, None], scales], -1))
        return query + tau[:, None] * self.output_scale * value

    def forward(self, tokens, geometry, query, tau, scales):
        return self.query_context(self.encode_support(tokens, geometry), query, tau, scales)


def normalization(tokens):
    values = np.asarray(tokens, np.float64).reshape(-1, tokens.shape[-1])
    mean, std = values.mean(0), values.std(0)
    # Constant channels are retained, not selected/dropped or projected.
    std[std < 1e-8] = 1.
    return mean.astype(np.float32), std.astype(np.float32)


def local_predictions(model, arrays, device, query=None, tau=None, batch_size=4096):
    query = arrays['query'] if query is None else np.asarray(query, np.float32)
    tau = np.linspace(0, 1, arrays['target'].shape[2], dtype=np.float32) if tau is None else np.asarray(tau, np.float32)
    n, q, _ = query.shape
    values = []
    with torch.no_grad():
        contexts = []
        for i in range(0, n, 256):
            contexts.append(model.encode_support(torch.as_tensor(arrays['tokens'][i:i+256], device=device),
                                                 torch.as_tensor(arrays['geometry'][i:i+256], device=device)))
        context = torch.cat(contexts)
        qs = torch.as_tensor(query, device=device)
        ts = torch.as_tensor(tau, device=device)
        scales = torch.as_tensor(arrays['scales'], device=device)
        for start in range(0, n*q*len(tau), batch_size):
            index = torch.arange(start, min(start+batch_size, n*q*len(tau)), device=device)
            i = index // (q*len(tau)); j = (index // len(tau)) % q; k = index % len(tau)
            values.append(model.query_context(context[i], qs[i, j], ts[k], scales[i]).cpu().numpy())
    return np.concatenate(values).reshape(n, q, len(tau), 3)


def predict_direct(model, arrays, device, **kwargs):
    return (local_predictions(model, arrays, device, **kwargs) * arrays['radius'][:, None, None, None]
            + arrays['origin'][:, None, None])


def compose_direct(model, arrays, device):
    n = len(arrays['origin']) // 2
    first, second = ({k: v[:n] for k, v in arrays.items()}, {k: v[n:] for k, v in arrays.items()})
    a = predict_direct(model, first, device)
    arrival = (a[:, :, -1] - second['origin'][:, None]) / second['radius'][:, None, None]
    b = predict_direct(model, second, device, query=arrival)
    return np.concatenate([a, b[:, :, 1:]], 2), dict(second_query_uses='predicted_first_stage_endpoint',
        predicted_arrival_outside_fraction=float((np.abs(arrival).sum(-1) > 1).mean()))


def fit_direct(arrays, settings, seed, device, on_progress=None):
    """Optimize training positions only; validation/test never enter this function."""
    torch.manual_seed(seed)
    mean, std = normalization(arrays['tokens'])
    delta = arrays['target'] - arrays['query'][:, :, None]
    output_scale = max(float(np.sqrt(np.mean(delta ** 2))), 1e-4)
    model = DirectFMTDecoder(mean, std, settings['hidden_width'], settings.get('support_blocks', 2),
                             settings.get('query_blocks', 4), output_scale, settings.get('small', False)).to(device)
    data = {k: torch.as_tensor(arrays[k], dtype=torch.float32, device=device)
            for k in ('tokens', 'geometry', 'query', 'scales', 'target')}
    n, q, t, _ = data['target'].shape
    optim = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=0.)
    start = time.perf_counter()
    curve = []
    generator = torch.Generator(device=device).manual_seed(seed + 113)
    probe = (torch.randint(n, (4096,), generator=generator, device=device),
             torch.randint(q, (4096,), generator=generator, device=device),
             torch.randint(1, t, (4096,), generator=generator, device=device))

    def measure(step, loss=None):
        model.eval()
        with torch.no_grad():
            if settings.get('stop_train_nrmse') is not None:
                prediction = local_predictions(model, arrays, device)
                error = prediction[:, :, 1:] - arrays['target'][:, :, 1:]
                fit_error = float(np.sqrt(np.mean(np.sum(error.astype(float)**2, -1))))
                population = 'all training particles and noninitial times'
            else:
                i, j, k = probe
                # Cache each observed support once during a diagnostic query.
                contexts = model.encode_support(data['tokens'], data['geometry'])
                prediction = model.query_context(contexts[i], data['query'][i, j], k.float()/(t-1), data['scales'][i])
                fit_error = float((prediction-data['target'][i, j, k]).square().sum(-1).mean().sqrt().cpu())
                population = 'fixed 4096 training queries, sampled before optimization'
        item = dict(step=step, elapsed_seconds=time.perf_counter()-start, train_position_nrmse=fit_error,
                    population=population, learning_rate=optim.param_groups[0]['lr'])
        if loss is not None:
            item['batch_scaled_mse'] = float(loss)
        curve.append(item)
        if on_progress:
            on_progress(item)
        model.train()
        return fit_error

    initial_error = measure(0)
    stop_reason = 'maximum_updates'
    total = settings['optimizer_steps']
    for step in range(1, total+1):
        phase = (step-1)/max(1, total-1)
        rate = settings['learning_rate'] * (.05+.95*.5*(1+math.cos(math.pi*phase)))
        for group in optim.param_groups:
            group['lr'] = rate
        # Several queries share each support encoding, so larger contexts are affordable.
        groups, queries = settings['bundle_batch'], settings['queries_per_bundle']
        i = torch.randint(n, (groups,), device=device)
        j = torch.randint(q, (groups, queries), device=device)
        k = torch.randint(1, t, (groups, queries), device=device)
        context = model.encode_support(data['tokens'][i], data['geometry'][i])
        pred = model.query_context(context[:, None].expand(-1, queries, -1).reshape(groups*queries, -1),
            data['query'][i[:, None], j].reshape(-1, 3), k.flatten().float()/(t-1),
            data['scales'][i, None].expand(-1, queries, -1).reshape(-1, 2))
        truth = data['target'][i[:, None], j, k].reshape(-1, 3)
        loss = ((pred-truth)/output_scale).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss; never hide failed fits')
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optim.step()
        if step % settings['probe_every'] == 0 or step == total:
            error = measure(step, loss.detach().cpu())
            threshold = settings.get('stop_train_nrmse')
            if threshold is not None and step >= settings.get('minimum_steps', 2000) and error <= threshold:
                stop_reason = 'preregistered_full_training_error_threshold'
                break
    model.eval()
    return model, dict(parameter_count=sum(p.numel() for p in model.parameters()),
        input_dimension=len(mean), projection='none', normalization='invertible per-channel affine inside network, fitted on training supports only',
        optimizer_steps=step, maximum_steps=total, initial_train_probe_nrmse=initial_error,
        final_train_probe_nrmse=curve[-1]['train_position_nrmse'], stop_reason=stop_reason,
        training_examples=n, query_particles_per_example=q, query_times=t,
        training_output_scale=output_scale, fit_seconds=time.perf_counter()-start, learning_curve=curve)
