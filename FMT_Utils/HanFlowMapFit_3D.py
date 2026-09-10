"""Fixed architecture with dense-query exposure and genuine time holdout, v1.1."""
from __future__ import annotations

import math
import time
import numpy as np
import torch
from FMT_Utils.FlowMapFit_3D import DirectFMTDecoder, normalization, local_predictions


def fit_dense(arrays, settings, seed, device, progress=None):
    torch.manual_seed(seed)
    n, q, full_times, _ = arrays['target'].shape
    indices = np.arange(2, full_times, 2)
    # No withheld temporal truth enters scaling, loss, stopping or progress probes.
    selected = arrays['target'][:, :, indices].copy()
    output_scale = max(float(np.sqrt(np.mean((selected - arrays['query'][:, :, None]) ** 2))), 1e-4)
    mean, std = normalization(arrays['tokens'])
    model = DirectFMTDecoder(mean, std, settings['hidden_width'], settings['support_blocks'],
                             settings['query_blocks'], output_scale).to(device)
    values = {k: torch.as_tensor(arrays[k], dtype=torch.float32, device=device)
              for k in ('tokens', 'geometry', 'query', 'scales')}
    target = torch.as_tensor(selected, dtype=torch.float32, device=device)
    tau = torch.as_tensor(indices / (full_times - 1), dtype=torch.float32, device=device)
    pairs_per_step = settings['bundle_batch'] * settings['queries_per_bundle']
    total = int(math.ceil(settings['equivalent_epochs'] * n * q * len(indices) / pairs_per_step))
    # Smoke tests can specify a fixed small budget, recorded explicitly.
    total = settings.get('smoke_steps', total)
    if total < 1:
        raise ValueError('Empty optimization budget')
    optim = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=0.)
    generator = torch.Generator(device=device).manual_seed(seed + 119)
    probe = tuple(torch.randint(size, (4096,), generator=generator, device=device) for size in (n, q, len(indices)))
    curve, start = [], time.perf_counter()

    def measure(step, loss=None):
        model.eval()
        with torch.no_grad():
            context = model.encode_support(values['tokens'], values['geometry'])
            i, j, k = probe
            estimate = model.query_context(context[i], values['query'][i, j], tau[k], values['scales'][i])
            error = (estimate - target[i, j, k]).square().sum(-1).mean().sqrt().item()
        record = dict(step=step, train_position_nrmse=error, elapsed_seconds=time.perf_counter() - start,
                      equivalent_exposure=step * pairs_per_step / (n * q * len(indices)),
                      population='fixed 4096 training queries at supervised even times only',
                      learning_rate=optim.param_groups[0]['lr'])
        if loss is not None:
            record['batch_scaled_mse'] = loss
        curve.append(record)
        if progress:
            progress(record)
        model.train()

    measure(0)
    for step in range(1, total + 1):
        rate = settings['learning_rate'] * (.05 + .95 * .5 * (1 + math.cos(math.pi * (step - 1) / max(1, total - 1))))
        for group in optim.param_groups:
            group['lr'] = rate
        b, nq = settings['bundle_batch'], settings['queries_per_bundle']
        i = torch.randint(n, (b,), device=device)
        j = torch.randint(q, (b, nq), device=device)
        k = torch.randint(len(indices), (b, nq), device=device)
        context = model.encode_support(values['tokens'][i], values['geometry'][i])
        predicted = model.query_context(context[:, None].expand(-1, nq, -1).reshape(b * nq, -1),
            values['query'][i[:, None], j].reshape(-1, 3), tau[k].flatten(),
            values['scales'][i, None].expand(-1, nq, -1).reshape(b * nq, -1))
        loss = ((predicted - target[i[:, None], j, k].reshape(-1, 3)) / output_scale).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite optimization loss')
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optim.step()
        if step % settings['probe_every'] == 0 or step == total:
            measure(step, float(loss.detach()))
    model.eval()
    return model, dict(parameter_count=sum(p.numel() for p in model.parameters()),
        input_dimension=len(mean), projection='none', optimizer_steps=total,
        training_examples=n, query_particles_per_example=q, supervised_time_indices=indices.tolist(),
        withheld_time_indices=np.arange(1, full_times, 2).tolist(), query_times=full_times,
        equivalent_epochs=total * pairs_per_step / (n * q * len(indices)),
        requested_equivalent_epochs=settings['equivalent_epochs'], input_scaling='training-support affine only',
        training_output_scale=output_scale, initial_train_probe_nrmse=curve[0]['train_position_nrmse'],
        final_train_probe_nrmse=curve[-1]['train_position_nrmse'], learning_curve=curve,
        fit_seconds=time.perf_counter() - start)
