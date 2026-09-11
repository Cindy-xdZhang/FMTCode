"""Task6 pnn_trans 1.1: fixed per-snapshot geometry, learned temporal compression.

Point-NN operators adapted from Renrui Zhang (MIT), upstream a85bfc365258a2c65a5f6ac289537b4d7a7cec0f.
See third_party/point_nn/LICENSE and docs/Task6_pnn_trans_protocol_1.1.md.
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from FMT_Utils.FlowMapData_3D import cross_offsets, write_json
from FMT_Utils.Task6DirectNeural_3D import check_geometry


def position_encoding(x, frequencies):
    """XYZ -> axis-major, frequency-major interleaved sine/cosine channels."""
    phase = x.unsqueeze(-1) * frequencies
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-3)


class SnapshotPointNN(nn.Module):
    """Zero parameters and no batch-dependent or running statistics.

    All seven material points are anchors; each uses all seven neighbors.
    Ordered anchor concatenation retains material-line correspondence.
    """
    def __init__(self, position_mean, position_scale, frequencies=8, alpha=1000., beta=1.):
        super().__init__()
        self.register_buffer('position_mean', torch.as_tensor(position_mean, dtype=torch.float32))
        self.register_buffer('position_scale', torch.as_tensor(position_scale, dtype=torch.float32))
        self.register_buffer('initial_frequencies', beta / alpha ** (torch.arange(frequencies).float()/frequencies))
        self.register_buffer('local_frequencies', beta / alpha ** (torch.arange(2*frequencies).float()/(2*frequencies)))
        self.output_dim = 7 * 18 * frequencies

    def forward(self, geometry):
        # Leading dimensions are arbitrary: [batch,time,7,3] or [snapshots,7,3].
        q = (geometry - self.position_mean) / self.position_scale
        initial = position_encoding(q, self.initial_frequencies)
        # i is the anchor, j its neighbor, with no temporal mixing here.
        dp = q.unsqueeze(-3) - q.unsqueeze(-2)
        df = initial.unsqueeze(-3) - initial.unsqueeze(-2)
        pstd = dp.std(dim=(-3, -2, -1), keepdim=True, unbiased=True)
        fstd = df.std(dim=(-3, -2, -1), keepdim=True, unbiased=True)
        relative = dp / (pstd + 1e-5)
        expanded = torch.cat((df/(fstd + 1e-5), initial.unsqueeze(-2).expand_as(df)), dim=-1)
        weights = position_encoding(relative, self.local_frequencies)
        weighted = (expanded + weights) * weights
        pooled = weighted.amax(-2) + weighted.mean(-2)
        # Upstream eval-mode, untouched BatchNorm has unit variance/affine.
        pooled = F.gelu(pooled / math.sqrt(1. + 1e-5))
        return torch.cat((initial, pooled), dim=-1).flatten(-2)


def fit_statistics(geometry):
    y = check_geometry(geometry).astype(np.float64)
    position_mean = y.mean(axis=(0, 1, 2))
    position_scale = max(float(np.sqrt(np.mean(np.sum((y-position_mean)**2, axis=-1)))), 1e-3)
    output_mean = y[:, :, 1:].mean(0)
    output_scale = max(float(np.sqrt(np.mean(np.sum((y[:, :, 1:]-output_mean)**2, axis=-1)))), 1e-3)
    return dict(position_mean=position_mean, position_scale=position_scale,
                output_mean=output_mean.reshape(651), output_scale=output_scale)


class DecoderBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                    nn.Dropout(dropout), nn.Linear(width, width))

    def forward(self, x):
        return x + .1*self.layers(x)


class PNNTransAutoencoder(nn.Module):
    """The decoder receives only a 192D latent, never the input or its tokens."""
    def __init__(self, statistics, arm='pnn_trans', width=128, heads=4, layers=3,
                 latent_dim=192, decoder_width=512, decoder_blocks=3, dropout=.05,
                 frequencies=8, alpha=1000., beta=1.):
        super().__init__()
        if arm not in ('pnn_trans', 'raw_trans'):
            raise ValueError(arm)
        self.arm, self.latent_dim = arm, latent_dim
        self.frontend = SnapshotPointNN(statistics['position_mean'], statistics['position_scale'],
                                       frequencies, alpha, beta)
        self.register_buffer('output_mean', torch.as_tensor(statistics['output_mean'], dtype=torch.float32))
        self.register_buffer('output_scale', torch.as_tensor(statistics['output_scale'], dtype=torch.float32))
        self.register_buffer('initial_seeds', torch.as_tensor(cross_offsets(), dtype=torch.float32))
        input_dim = self.frontend.output_dim if arm == 'pnn_trans' else 21
        self.input_projection = nn.Linear(input_dim, width)
        self.time_embedding = nn.Parameter(torch.randn(1, 32, width)*.02)
        # Construct independently; TransformerEncoder copies share initial values otherwise.
        self.temporal_layers = nn.ModuleList([nn.TransformerEncoderLayer(width, heads, 4*width,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True) for _ in range(layers)])
        self.latent_dropout = nn.Dropout(dropout)
        self.compress = nn.Linear(32*width, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, decoder_width),
            *[DecoderBlock(decoder_width, dropout) for _ in range(decoder_blocks)],
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(decoder_width, 651))
        # The initial output is the training mean, not a fixed inverse or PCA reconstruction.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def snapshot_features(self, geometry):
        points = geometry.transpose(1, 2)
        if self.arm == 'pnn_trans':
            return self.frontend(points)
        return ((points-self.frontend.position_mean)/self.frontend.position_scale).flatten(-2)

    def encode_features(self, features):
        x = self.input_projection(features) + self.time_embedding
        for layer in self.temporal_layers:
            x = layer(x)
        return self.compress(self.latent_dropout(x.flatten(1)))

    def encode(self, geometry):
        return self.encode_features(self.snapshot_features(geometry))

    def decode(self, latent):
        later = (self.decoder(latent)*self.output_scale + self.output_mean).reshape(-1, 7, 31, 3)
        initial = self.initial_seeds[None, :, None].expand(len(latent), -1, -1, -1)
        return torch.cat((initial, later), dim=2)

    def forward(self, geometry):
        return self.decode(self.encode(geometry))


def audit_model(model):
    assert sum(p.numel() for p in model.frontend.parameters()) == 0
    assert all(p.requires_grad for p in model.parameters())
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
    allowed = {'output_mean', 'output_scale', 'initial_seeds', 'frontend.position_mean',
               'frontend.position_scale', 'frontend.initial_frequencies', 'frontend.local_frequencies'}
    assert set(dict(model.named_buffers())) == allowed
    return dict(arm=model.arm, frontend_parameters=0, trainable_parameters=sum(p.numel() for p in model.parameters()),
                input_shape=[7, 32, 3], snapshot_dimension=model.input_projection.in_features,
                latent_dimension=model.latent_dim, decoder_input='latent only',
                analytic_inverse=False, pca_dependency=False, raw_skip=False,
                autoencoder='deterministic; no variational objective')


@torch.no_grad()
def cached_features(model, geometry, device, batch=128):
    model.eval()
    return torch.cat([model.snapshot_features(torch.as_tensor(geometry[i:i+batch], device=device))
                      for i in range(0, len(geometry), batch)])


@torch.no_grad()
def score(model, features, geometry, batch=256, zero_latent=False):
    model.eval()
    total = 0.
    for i in range(0, len(features), batch):
        z = model.encode_features(features[i:i+batch])
        prediction = model.decode(torch.zeros_like(z) if zero_latent else z)
        truth = torch.as_tensor(geometry[i:i+batch], device=features.device)
        total += (prediction[:, :, 1:].double()-truth[:, :, 1:].double()).square().sum().item()
    return math.sqrt(total/(len(features)*7*31))


@torch.no_grad()
def predict(model, geometry, device, batch=128):
    model.eval()
    return np.concatenate([model(torch.as_tensor(geometry[i:i+batch], device=device)).cpu().numpy()
                           for i in range(0, len(geometry), batch)])


def train_model(y, vy, arm, candidate, training, seed, device, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    check_geometry(vy)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    statistics = fit_statistics(y)
    model = PNNTransAutoencoder(statistics, arm=arm, dropout=candidate['dropout'], **training['architecture']).to(device)
    structure = audit_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate['learning_rate'], weight_decay=candidate['weight_decay'])
    x, vx = cached_features(model, y, device), cached_features(model, vy, device)
    target = torch.as_tensor(y, device=device)
    rng = np.random.default_rng(seed)
    batch = min(training['batch_size'], len(y))
    batches = math.ceil(len(y)/batch)
    best, best_step, best_state = float('inf'), None, None
    start, curve = time.monotonic(), []

    def record(step, loss=None):
        nonlocal best, best_step, best_state
        val, fit = score(model, vx, vy), score(model, x, y)
        if not np.isfinite([val, fit]).all():
            raise FloatingPointError('Nonfinite RMSE')
        row = dict(step=step, train_rmse_r=fit, validation_rmse_r=val,
                   loss_normalized=loss, seconds=time.monotonic()-start)
        curve.append(row)
        if step > 0 and val < best:
            best, best_step = val, step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        write_json(folder/'progress.json', row)
        print(dict(arm=arm, candidate=candidate['id'], **row), flush=True)

    record(0)
    exposed = 0
    for step in range(training['updates']):
        if step % batches == 0:
            order = rng.permutation(len(y))
        ix = torch.as_tensor(order[(step % batches)*batch:((step % batches)+1)*batch], device=device)
        exposed += len(ix)
        warmup = min(1., (step+1)/max(1, training['warmup_updates']))
        rate = candidate['learning_rate']*warmup*(.1+.45*(1+math.cos(math.pi*step/max(1, training['updates']-1))))
        for group in optimizer.param_groups:
            group['lr'] = rate
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model.decode(model.encode_features(x[ix]))
        loss = (prediction[:, :, 1:]-target[ix, :, 1:]).square().sum(-1).mean()/model.output_scale.square()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'])
        optimizer.step()
        if (step+1) % training['probe_every'] == 0 or step+1 == training['updates']:
            record(step+1, loss.detach().item())
    assert best_state is not None
    model.load_state_dict(best_state)
    result = dict(arm=arm, candidate=candidate, seed=seed, structure=structure, train_samples=len(y),
        validation_samples=len(vy), statistics_train_samples=len(y), updates=training['updates'],
        train_examples_exposed=exposed, selected_step=best_step, validation_rmse_r=best,
        train_rmse_r=score(model, x, y), train_zero_latent_rmse_r=score(model, x, y, zero_latent=True),
        curve=curve, seconds=time.monotonic()-start, model_files='none; best state RAM only',
        target_scale=statistics['output_scale'], position_scale=statistics['position_scale'],
        selection='minimum full validation RMSE at positive steps; no test access')
    write_json(folder/'fit.json', result)
    return model, result
