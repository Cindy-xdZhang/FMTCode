"""Octahedral-equivariant SIREN-VAE for 7-line 3D pathline primitives.

A 3D port of the 2D ``D8``-equivariant SIREN-VAE in
``PyflowVis/FMT_Utils/siren_vae.py`` and ``FMT_Clustering_SIRENVAE.py``, keeping
all four of that method's default refinements:

1. encoder learning rate scaled down relative to the decoder's;
2. channel-balanced reconstruction loss (centre weight + one shared neighbour
   weight, which is what keeps it group-invariant);
3. label-free checkpoint selection by Davies-Bouldin on ``z_inv``;
4. decoder-only fine-tune on the restored, frozen encoder.

The latent splits as ``z = [z_inv | z_eq]``.  A positive-pairs-only contrastive
loss pulls ``z_inv`` together across octahedral views; ``z_eq`` is free and
absorbs orientation so reconstruction stays possible.  Because that contrastive
loss has no negatives its optimum is ``z_inv = const``, so a VICReg variance and
covariance term puts a floor under it.

3D-specific choices are documented in ``docs/octahedral_equivariance_3d.md``.
``z_eq`` defaults to 12 rather than 8 because orientation in 3D is a
3-dimensional manifold acted on by 48 group elements, against 1 dimension and
16 elements in 2D.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import davies_bouldin_score
from torch import nn
from torch.nn import functional as F

from FMT_Utils.OctahedralGroup_3D import (
    LINE_COUNT, apply_group, apply_norm_stats_oh, channel_balance_weights_oh,
    group_tensors, normalize_signal_oh, random_group,
)

VECTOR_DIM = 3
READOUTS = ("gmean", "plain", "gmax", "gmeanmax")
# `gmean` -- mean of z_inv over the 48 group elements -- is the default
# readout: it is exactly O_h-invariant by construction and led the
# confirmation table in versions 1.3 and 1.4.
DEFAULT_READOUT = "gmean"


# ---------------------------------------------------------------------------
# SIREN decoder
# ---------------------------------------------------------------------------

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, w0=30.0):
        super().__init__()
        self.w0 = float(w0)
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            bound = (1.0 / in_features) if is_first else (math.sqrt(6.0 / in_features) / w0)
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


class SIRENDecoder3D(nn.Module):
    """Maps ``(t in [0,1], z)`` to the signal at that time."""

    def __init__(self, z_dim, out_dim, hidden_dim=128, num_layers=4,
                 w0_first=30.0, w0=30.0):
        super().__init__()
        layers = [SineLayer(1 + z_dim, hidden_dim, is_first=True, w0=w0_first)]
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, w0=w0))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / w0
            self.head.weight.uniform_(-bound, bound)
            self.head.bias.zero_()

    def forward(self, t, z):
        steps = t.shape[1]
        return self.head(self.net(torch.cat([t, z.unsqueeze(1).expand(-1, steps, -1)], dim=-1)))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def _pick_groups(channels, target=8):
    groups = min(target, channels)
    while channels % groups:
        groups -= 1
    return max(groups, 1)


class ResConvBlock1D(nn.Module):
    """Pre-activation residual 1D-conv block with optional AvgPool/2."""

    def __init__(self, in_ch, out_ch, groups=8, downsample=True):
        super().__init__()
        self.norm1 = nn.GroupNorm(_pick_groups(in_ch, groups), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_pick_groups(out_ch, groups), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()
        self.pool = nn.AvgPool1d(2, ceil_mode=True) if downsample else nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return self.pool(h + self.skip(x))


class ResTransformerEncoder3D(nn.Module):
    """Two residual conv stages (each halving time) then a CLS transformer."""

    def __init__(self, in_channels, seq_len, z_dim, conv_channels=(64, 128),
                 d_model=128, nhead=4, num_layers=3, dim_ff=256, dropout=0.0):
        super().__init__()
        first, second = conv_channels
        self.block1 = ResConvBlock1D(in_channels, first, downsample=True)
        self.block2 = ResConvBlock1D(first, second, downsample=True)
        steps = seq_len
        for _ in range(2):
            steps = (steps + 1) // 2
        self.proj = nn.Conv1d(second, d_model, 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, steps + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.mu_head = nn.Linear(d_model, z_dim)
        self.logvar_head = nn.Linear(d_model, z_dim)

    def forward(self, x):
        h = self.proj(self.block2(self.block1(x))).transpose(1, 2)
        h = torch.cat([self.cls_token.expand(h.shape[0], -1, -1), h], dim=1)
        h = self.final_norm(self.transformer(h + self.pos_embed[:, :h.shape[1]]))
        return self.mu_head(h[:, 0]), self.logvar_head(h[:, 0])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PureTransformerEncoder3D(nn.Module):
    """CLS transformer straight on the time axis -- no convolutional front end.

    ``ResTransformerEncoder3D`` already ends in a transformer, but two residual
    conv stages halve time twice first, so attention only ever sees T/4 tokens
    and the local mixing is done by convolution.  This variant embeds every time
    step as its own token, so all the mixing is attention.
    """

    def __init__(self, in_channels, seq_len, z_dim, d_model=128, nhead=4,
                 num_layers=3, dim_ff=256, dropout=0.0):
        super().__init__()
        self.embed = nn.Linear(in_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.mu_head = nn.Linear(d_model, z_dim)
        self.logvar_head = nn.Linear(d_model, z_dim)

    def forward(self, x):
        h = self.embed(x.transpose(1, 2))                       # [B, T, d_model]
        h = torch.cat([self.cls_token.expand(h.shape[0], -1, -1), h], dim=1)
        h = self.final_norm(self.transformer(h + self.pos_embed[:, :h.shape[1]]))
        return self.mu_head(h[:, 0]), self.logvar_head(h[:, 0])


class OctahedralSirenVAE3D(nn.Module):
    """VAE over ``[B, 7, T, 3]`` primitives with a split ``[z_inv | z_eq]`` latent."""

    def __init__(self, steps, z_inv_dim=16, z_eq_dim=12, conv_channels=(64, 128),
                 d_model=128, nhead=4, trans_layers=3, dim_ff=256,
                 dec_hidden=128, dec_layers=4, w0=30.0, w0_first=30.0,
                 encoder_kind="conv"):
        super().__init__()
        self.steps = int(steps)
        self.z_inv_dim = int(z_inv_dim)
        self.z_eq_dim = int(z_eq_dim)
        self.z_dim = self.z_inv_dim + self.z_eq_dim
        self.channels = LINE_COUNT * VECTOR_DIM
        self.encoder_kind = encoder_kind
        if encoder_kind == "pure":
            self.encoder = PureTransformerEncoder3D(
                self.channels, self.steps, self.z_dim, d_model=d_model,
                nhead=nhead, num_layers=trans_layers, dim_ff=dim_ff,
            )
        elif encoder_kind == "conv":
            self.encoder = ResTransformerEncoder3D(
                self.channels, self.steps, self.z_dim, conv_channels=conv_channels,
                d_model=d_model, nhead=nhead, num_layers=trans_layers, dim_ff=dim_ff,
            )
        else:
            raise ValueError(f"encoder_kind must be 'conv' or 'pure', got {encoder_kind!r}")
        self.decoder = SIRENDecoder3D(
            self.z_dim, self.channels, hidden_dim=dec_hidden,
            num_layers=dec_layers, w0_first=w0_first, w0=w0,
        )
        self.register_buffer(
            "t_grid", torch.linspace(0.0, 1.0, self.steps).view(1, self.steps, 1),
            persistent=False,
        )

    def _flatten(self, signal):
        batch, lines, steps, comps = signal.shape
        return signal.permute(0, 1, 3, 2).reshape(batch, lines * comps, steps)

    def encode(self, signal):
        return self.encoder(self._flatten(signal))

    def decode(self, z):
        decoded = self.decoder(self.t_grid.expand(z.shape[0], -1, -1), z)
        return decoded.view(z.shape[0], self.steps, LINE_COUNT, VECTOR_DIM).permute(0, 2, 1, 3)

    def forward(self, signal, sample=False):
        mu, logvar = self.encode(signal)
        z = mu
        if sample and self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder = sum(p.numel() for p in self.encoder.parameters())
        return {"total": int(total), "trainable": int(trainable),
                "encoder": int(encoder), "decoder": int(total - encoder)}


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def multi_view_contrastive_loss(views):
    """Mean ``1 - cosine`` over all pairs of views of the same instance."""
    total, count = views[0].new_zeros(()), 0
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            total = total + (1.0 - F.cosine_similarity(views[i], views[j], dim=-1)).mean()
            count += 1
    return total / max(count, 1)


def _off_diagonal(matrix):
    n = matrix.shape[0]
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def zinv_variance_covariance(views, target_std=0.5, eps=1e-4):
    """VICReg variance floor and covariance decorrelation on ``z_inv``.

    The contrastive term above has no negatives, so its global optimum is the
    degenerate ``z_inv = const``.  These two terms are what stop the invariance
    objective from being bought by collapsing the code.
    """
    var_total, cov_total = views[0].new_zeros(()), views[0].new_zeros(())
    for z in views:
        count, dim = z.shape
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        var_total = var_total + torch.mean(F.relu(target_std - std))
        centred = z - z.mean(dim=0)
        cov = (centred.T @ centred) / max(count - 1, 1)
        cov_total = cov_total + _off_diagonal(cov).square().sum() / dim
    return var_total / len(views), cov_total / len(views)


def vae_oh_loss(recon, target, mu, logvar, contrastive, beta=1e-4,
                lambda_contrast=15.0, channel_weights=None):
    if channel_weights is not None:
        recon_loss = (channel_weights * (recon - target).pow(2)).mean()
    else:
        recon_loss = F.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl + lambda_contrast * contrastive
    return total, recon_loss.detach(), kl.detach(), contrastive.detach()


# ---------------------------------------------------------------------------
# Group-pooled readouts
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_readouts(model, signal_n, matrices, permutation, batch_size=2048,
                    readouts=READOUTS, element_block=8):
    """Encode normalised signals into every requested invariant readout.

    ``plain``  : ``z_inv`` of the primitive as cached -- invariant only to the
                 extent the contrastive loss succeeded.
    ``gmean``  : ``mean_g z_inv(g . x)`` -- exactly ``O_h``-invariant by
                 construction, whatever the encoder learned.
    ``gmax``   : ``max_g z_inv(g . x)`` elementwise -- also exactly invariant,
                 and unlike the mean it does not cancel orbit structure.
    ``gmeanmax``: the two concatenated.

    Reporting ``plain`` beside the pooled readouts separates "the method works"
    from "the contrastive loss converged", which the 2D setup cannot do.  The
    group is applied to the already-normalised signal, which is exact because
    ``apply_norm_stats_oh`` commutes with the action (see
    ``tests/test_octahedral_group_3d.py``).
    """
    model.eval()
    wanted = set(readouts)
    needs_group = bool(wanted & {"gmean", "gmax", "gmeanmax"})
    parts = {name: [] for name in readouts}
    for start in range(0, len(signal_n), int(batch_size)):
        chunk = signal_n[start:start + int(batch_size)]
        mu, _ = model.encode(chunk)
        plain = mu[:, :model.z_inv_dim]
        if "plain" in wanted:
            parts["plain"].append(plain.float().cpu())
        if not needs_group:
            continue
        # The encoder is launch-bound, not compute-bound, so several group
        # elements are stacked into one forward pass.  Every layer here
        # (GroupNorm, LayerNorm, attention) is per-sample, so stacking is
        # numerically identical to encoding the elements one at a time.
        running_sum, running_max = None, None
        rows = len(chunk)
        for first in range(0, len(matrices), int(element_block)):
            indices = range(first, min(first + int(element_block), len(matrices)))
            stacked = torch.cat([
                apply_group(chunk, torch.full((rows,), index, dtype=torch.long,
                                              device=chunk.device),
                            matrices, permutation)
                for index in indices
            ], dim=0)
            z = model.encode(stacked)[0][:, :model.z_inv_dim].view(len(indices), rows, -1)
            block_sum = z.sum(dim=0)
            block_max = z.max(dim=0).values
            running_sum = block_sum if running_sum is None else running_sum + block_sum
            running_max = block_max if running_max is None else torch.maximum(running_max, block_max)
        mean = (running_sum / len(matrices)).float().cpu()
        maximum = running_max.float().cpu()
        if "gmean" in wanted:
            parts["gmean"].append(mean)
        if "gmax" in wanted:
            parts["gmax"].append(maximum)
        if "gmeanmax" in wanted:
            parts["gmeanmax"].append(torch.cat((mean, maximum), dim=1))
    return {name: torch.cat(values).numpy().astype(np.float32)
            for name, values in parts.items() if values}


def l2_normalize(values, eps=1e-8):
    """Row-wise L2 normalisation, as the 2D method uses before k-means."""
    values = np.asarray(values, dtype=np.float32)
    scale = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(scale, eps)


def davies_bouldin_selection_score(features, labels):
    """Label-free checkpoint criterion; higher is better, so Davies-Bouldin is negated."""
    if len(np.unique(labels)) < 2:
        return float("-inf")
    return -float(davies_bouldin_score(np.asarray(features, dtype=np.float64), labels))


@torch.no_grad()
def reconstruction_mse(model, signal_n, channel_weights=None, batch_size=4096):
    """Mean reconstruction error of ``signal_n`` under the current model.

    Reported on held-out ordinals so that reconstruction is a measurement of the
    primitive, not a training-set fit -- the 2D README's caveat that
    "reconstruction MSE is a training-set fit" applies directly here.
    """
    # Restore the caller's mode rather than forcing train().  Forcing it flipped
    # the encoder out of the eval() state the decoder fine-tune sets, and
    # nn.TransformerEncoder takes a different fast path in train vs eval, which
    # showed up as a spurious ~2e-4 "z_inv drift" during the fine-tune even
    # though the encoder parameters are provably frozen (requires_grad False and
    # absent from the fine-tune optimizer).
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(signal_n), int(batch_size)):
        chunk = signal_n[start:start + int(batch_size)]
        mu, _ = model.encode(chunk)
        recon = model.decode(mu)
        if channel_weights is not None:
            error = (channel_weights * (recon - chunk).pow(2)).mean()
        else:
            error = F.mse_loss(recon, chunk)
        total += float(error) * len(chunk); count += len(chunk)
    if was_training:
        model.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fit_octvae(signal_train, signal_validation, settings, seed, device,
               group="oh", progress=None, signal_recon_validation=None,
               signal_recon_train_probe=None):
    """Train one model and return it restored to its selected checkpoint.

    Returns ``(model, stats, history)``.  ``stats`` are the frozen train-only
    normalisation statistics, which callers must reuse for every later encode.

    ``progress`` is called as ``progress(step, total_steps, history, model,
    stats)`` after every evaluation, which is what the selection diagnostic
    uses to probe the model mid-training without duplicating this loop.

    Checkpoint selection uses Davies-Bouldin on the validation ``z_inv`` with
    k-means labels fitted on the training ``z_inv`` -- no labels anywhere, so
    the confirmation slices stay frozen.  Freezing the encoder pins ``z_inv``
    exactly, so restoring epoch ``E`` and stopping at ``E`` are the same model;
    the decoder fine-tune then refits only the decoder, which cannot move
    ``z_inv``.
    """
    from sklearn.cluster import KMeans

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    generator = torch.Generator(device=device).manual_seed(int(seed))

    matrices, permutation = group_tensors(group, device=device)
    train = torch.as_tensor(signal_train, dtype=torch.float32, device=device)
    validation = torch.as_tensor(signal_validation, dtype=torch.float32, device=device)

    train_n, stats = normalize_signal_oh(train, settings["clip_sigma"])
    validation_n = apply_norm_stats_oh(validation, stats, settings["clip_sigma"])
    weights = (channel_balance_weights_oh(train_n)
               if settings["recon_channel_balance"] else None)

    model = OctahedralSirenVAE3D(
        steps=train.shape[2], z_inv_dim=settings["z_inv_dim"],
        z_eq_dim=settings["z_eq_dim"], dec_hidden=settings["dec_hidden"],
        dec_layers=settings["dec_layers"], w0=settings["w0"],
        w0_first=settings["w0_first"],
    ).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),
         "lr": settings["lr"] * settings["encoder_lr_scale"]},
        {"params": model.decoder.parameters(), "lr": settings["lr"]},
    ], weight_decay=settings["weight_decay"])
    total_steps = int(settings["steps"])
    # Linear LR warmup then cosine decay.  `lr_warmup_steps` is a genuine
    # learning-rate ramp and must not be confused with `selection_warmup`, which
    # only gates when a checkpoint may first be selected.  LambdaLR multiplies
    # each group's base lr, so the encoder/decoder ratio is preserved.
    lr_warmup = int(settings.get("lr_warmup_steps", 0))

    def _lr_scale(step):
        if lr_warmup > 0 and step < lr_warmup:
            return float(step + 1) / float(lr_warmup)
        progress = (step - lr_warmup) / max(total_steps - lr_warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_scale)

    batch = int(settings["batch_size"])
    history = {"step": [], "loss": [], "recon": [], "kl": [], "contrast": [],
               "zinv_std": [], "selection": []}
    best = {"score": float("-inf"), "step": -1,
            "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}

    # Fixed random subsample for the selection k-means.  Taking the first rows
    # instead would draw them from the earliest timeslices only, because the
    # training tensor is concatenated in slice order.
    selection_index = torch.randperm(
        len(train_n), device=device, generator=generator
    )[:int(settings["selection_subsample"])]

    def selection_score():
        model.eval()
        with torch.no_grad():
            train_z = l2_normalize(
                model.encode(train_n[selection_index])[0]
                [:, :model.z_inv_dim].float().cpu().numpy())
            validation_z = l2_normalize(
                model.encode(validation_n)[0][:, :model.z_inv_dim].float().cpu().numpy())
        kmeans = KMeans(n_clusters=2, random_state=int(seed),
                        n_init=settings["kmeans_n_init"]).fit(train_z)
        model.train()
        return davies_bouldin_selection_score(validation_z, kmeans.predict(validation_z))

    model.train()
    step = 0
    order = torch.randperm(len(train_n), device=device, generator=generator)
    cursor = 0
    while step < total_steps:
        if cursor + batch > len(order):
            order = torch.randperm(len(train_n), device=device, generator=generator)
            cursor = 0
        index = order[cursor:cursor + batch]
        cursor += batch

        original = train_n[index]
        rows = len(index)
        # One stacked encoder pass over [original, aug1, aug2] instead of three
        # separate passes.  Identical arithmetic -- every normalisation in the
        # encoder is per-sample -- but ~3x fewer kernel launches on a workload
        # that is launch-bound rather than compute-bound.
        augmented = [
            apply_norm_stats_oh(
                random_group(train[index], matrices, permutation,
                             generator=generator, exclude_identity=True)[0],
                stats, settings["clip_sigma"])
            for _ in range(2)
        ]
        stacked = torch.cat([original, *augmented], dim=0)
        mu_all, logvar_all = model.encode(stacked)
        z_inv = mu_all[:rows, :model.z_inv_dim]
        views = [mu_all[i * rows:(i + 1) * rows, :model.z_inv_dim] for i in range(3)]
        contrastive = multi_view_contrastive_loss(views)
        if settings.get("recon_augmented", False):
            # Reconstruct every view, not just the un-augmented one.  This forces
            # z_eq to carry the orientation of the whole orbit rather than only
            # the orientation the primitive happened to be cached in; the decoder
            # can no longer ignore the group.
            recon, target = model.decode(mu_all), stacked
            mu, logvar = mu_all, logvar_all
        else:
            mu, logvar = mu_all[:rows], logvar_all[:rows]
            recon, target = model.decode(mu), original

        loss, recon_value, kl_value, contrast_value = vae_oh_loss(
            recon, target, mu, logvar, contrastive, beta=settings["beta"],
            lambda_contrast=settings["lambda_contrast"], channel_weights=weights,
        )
        if settings["zinv_var_weight"] > 0 or settings["zinv_cov_weight"] > 0:
            variance, covariance = zinv_variance_covariance(
                views, target_std=settings["zinv_target_std"])
            loss = (loss + settings["zinv_var_weight"] * variance
                    + settings["zinv_cov_weight"] * covariance)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % int(settings["eval_every"]) == 0 or step == total_steps:
            score = (selection_score() if step >= int(settings["selection_warmup"])
                     else float("-inf"))
            history["step"].append(step)
            history["loss"].append(float(loss.item()))
            history["recon"].append(float(recon_value))
            history["kl"].append(float(kl_value))
            history["contrast"].append(float(contrast_value))
            history["zinv_std"].append(float(z_inv.detach().std(dim=0).mean()))
            history["selection"].append(None if score == float("-inf") else score)
            if score > best["score"]:
                best = {"score": score, "step": step,
                        "state": {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}}
            if progress is not None:
                progress(step, total_steps, history, model, stats)

    model.load_state_dict(best["state"])
    history["selected_step"] = best["step"]
    history["selected_score"] = best["score"]

    def _to_normalised(values):
        if values is None:
            return None
        return apply_norm_stats_oh(
            torch.as_tensor(values, dtype=torch.float32, device=device),
            stats, settings["clip_sigma"])

    recon_train_probe_n = _to_normalised(signal_recon_train_probe)
    recon_validation_n = None
    if signal_recon_validation is not None:
        recon_validation_n = apply_norm_stats_oh(
            torch.as_tensor(signal_recon_validation, dtype=torch.float32, device=device),
            stats, settings["clip_sigma"])
        history["recon_heldout_pre_finetune"] = reconstruction_mse(
            model, recon_validation_n, weights)
        history["recon_train_pre_finetune"] = reconstruction_mse(model, train_n, weights)
        if recon_train_probe_n is not None:
            history["recon_trainprobe_pre_finetune"] = reconstruction_mse(
                model, recon_train_probe_n, weights)

    finetune_steps = int(settings["decoder_finetune_steps"])
    if finetune_steps > 0:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        model.encoder.eval()
        finetune_optimizer = torch.optim.AdamW(
            model.decoder.parameters(), lr=settings["decoder_finetune_lr"],
            weight_decay=settings["weight_decay"])
        finetune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            finetune_optimizer, T_max=finetune_steps)
        with torch.no_grad():
            reference_z = model.encode(train_n[:2048])[0][:, :model.z_inv_dim].clone()
        curve = {"step": [], "recon_in_training": [], "recon_test": []}

        def _record(step_index):
            entry_a = (reconstruction_mse(model, recon_train_probe_n, weights)
                       if recon_train_probe_n is not None else float("nan"))
            entry_b = (reconstruction_mse(model, recon_validation_n, weights)
                       if recon_validation_n is not None else float("nan"))
            curve["step"].append(int(step_index))
            curve["recon_in_training"].append(entry_a)
            curve["recon_test"].append(entry_b)

        probe_every = int(settings.get("finetune_probe_every", 100))
        _record(0)
        for finetune_step in range(finetune_steps):
            if cursor + batch > len(order):
                order = torch.randperm(len(train_n), device=device, generator=generator)
                cursor = 0
            index = order[cursor:cursor + batch]
            cursor += batch
            original = train_n[index]
            with torch.no_grad():
                mu, _ = model.encode(original)
            recon = model.decode(mu)
            if weights is not None:
                recon_loss = (weights * (recon - original).pow(2)).mean()
            else:
                recon_loss = F.mse_loss(recon, original)
            finetune_optimizer.zero_grad(set_to_none=True)
            recon_loss.backward()
            nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            finetune_optimizer.step()
            finetune_scheduler.step()
            if probe_every > 0 and (finetune_step + 1) % probe_every == 0:
                _record(finetune_step + 1)
        history["finetune_curve"] = curve
        with torch.no_grad():
            drift = float((model.encode(train_n[:2048])[0][:, :model.z_inv_dim]
                           - reference_z).abs().max())
        history["finetune_zinv_drift"] = drift
        history["finetune_recon"] = float(recon_loss.item())
        if recon_validation_n is not None:
            history["recon_heldout_post_finetune"] = reconstruction_mse(
                model, recon_validation_n, weights)
            history["recon_train_post_finetune"] = reconstruction_mse(model, train_n, weights)
            if recon_train_probe_n is not None:
                history["recon_trainprobe_post_finetune"] = reconstruction_mse(
                    model, recon_train_probe_n, weights)
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(True)

    return model, stats, history
