"""IcoVAE coreline classification on the re-seeded Task6_CorelineDataset_2.7_share stars.

Trains on the pooled training frames of every scene and evaluates on every test
frame of every scene, which is what the task asks for.  The cache holds a 49-line
star (centre + 12 icosahedron vertices at 1h, 2h, 4h, 8h); ``--shells`` selects
any single radius or any multi-shell combination out of it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from FMT_Utils.IcoVAE_3D import (  # noqa: E402
    IcosahedralSirenVAE3D, apply_norm_stats_ico, build_signal_ico,
    channel_balance_weights_ico, multi_view_contrastive_loss, normalize_signal_ico,
    vae_ico_loss, zinv_variance_covariance)
from FMT_Utils.IcosahedralGroup_3D import (  # noqa: E402
    NEIGHBOUR_COUNT, VECTOR_DIM, apply_group_shells, group_tensors, icosahedron_vertices)

RADII = (1.0, 2.0, 4.0, 8.0)
TASK6_ROOT = os.environ.get(
    "TASK6_ROOT", "/home/cheny1a/data/flowData3D/Task6_CorelineDataset_2.7_share")
_BOUNDS = {}


def _frame_bounds(key):
    """Physical bounds of a frame, read once from the dataset package."""
    if key not in _BOUNDS:
        if TASK6_ROOT not in sys.path:
            sys.path.insert(0, TASK6_ROOT)
        from task6_fields import Dataset
        ds = _BOUNDS.setdefault("__ds__", Dataset(TASK6_ROOT))
        b = ds.frame(key).bounds
        _BOUNDS[key] = (b[:, 0].copy(), b[:, 1].copy())
    return _BOUNDS[key]


def so3_shells(signal, generator=None):
    """Continuous SO(3) view for a multi-shell star (repo helper is single-shell)."""
    rows, lines = signal.shape[0], signal.shape[1]
    shells = (lines - 1) // NEIGHBOUR_COUNT
    noise = torch.randn(rows, 3, 3, device=signal.device, dtype=signal.dtype, generator=generator)
    q, r = torch.linalg.qr(noise)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]
    flip = torch.where(torch.linalg.det(q) < 0, -1.0, 1.0)
    rot = torch.cat((q[:, :, :2], q[:, :, 2:] * flip[:, None, None]), dim=-1)
    d = torch.as_tensor(icosahedron_vertices(), device=signal.device, dtype=signal.dtype)
    gather = (torch.einsum("nji,kj->nki", rot, d) @ d.T).argmax(-1)
    pieces = [signal[:, :1]]
    for s in range(shells):
        start = 1 + NEIGHBOUR_COUNT * s
        pieces.append(torch.gather(
            signal[:, start:start + NEIGHBOUR_COUNT], 1,
            gather[:, :, None, None].expand(-1, -1, signal.shape[2], VECTOR_DIM)))
    return torch.einsum("nij,nktj->nkti", rot, torch.cat(pieces, dim=1))


def so3_snap_shells(signal, matrices, permutation, generator=None):
    """Continuous SO(3) view whose channel map is always a valid permutation.

    ``random_so3`` assigns channels by nearest icosahedral vertex, which is not
    bijective for ~31% of rotations (see Analyze_SO3_ChannelMap_1_1.py): two
    channels claim one vertex, so a neighbour is duplicated and another dropped.
    The only structure-preserving permutations of the twelve vertices are the
    group's own, so the best valid channel map for a rotation R is the one
    induced by the nearest group element, found by maximising tr(R^T G).  The
    vectors are still rotated by R exactly.
    """
    rows, lines = signal.shape[0], signal.shape[1]
    shells = (lines - 1) // NEIGHBOUR_COUNT
    noise = torch.randn(rows, 3, 3, device=signal.device, dtype=signal.dtype, generator=generator)
    q, r = torch.linalg.qr(noise)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]
    flip = torch.where(torch.linalg.det(q) < 0, -1.0, 1.0)
    rot = torch.cat((q[:, :, :2], q[:, :, 2:] * flip[:, None, None]), dim=-1)
    mats = matrices.to(signal.device).to(signal.dtype)
    # tr(R^T G) over every group element; the maximiser is the nearest rotation
    nearest = torch.einsum("nij,kij->nk", rot, mats).argmax(-1)
    gather = permutation.to(signal.device)[nearest]
    pieces = [signal[:, :1]]
    for s in range(shells):
        start = 1 + NEIGHBOUR_COUNT * s
        pieces.append(torch.gather(
            signal[:, start:start + NEIGHBOUR_COUNT], 1,
            gather[:, :, None, None].expand(-1, -1, signal.shape[2], VECTOR_DIM)))
    return torch.einsum("nij,nktj->nkti", rot, torch.cat(pieces, dim=1))


def views_of(signal_raw, stats, clip_sigma, matrices, permutation, mode, generator, count=2):
    shells = (signal_raw.shape[1] - 1) // NEIGHBOUR_COUNT
    out = []
    for i in range(count):
        choice = mode if mode != "both" else ("ih" if i % 2 == 0 else "so3")
        if choice == "so3":
            aug = so3_shells(signal_raw, generator)
        elif choice == "so3snap":
            aug = so3_snap_shells(signal_raw, matrices, permutation, generator)
        else:
            el = torch.randint(1, len(matrices), (signal_raw.shape[0],),
                               device=signal_raw.device, generator=generator)
            aug = apply_group_shells(signal_raw, el, matrices, permutation, shells)
        out.append(apply_norm_stats_ico(aug, stats, clip_sigma))
    return out


def load(cache, role, shells, scenes=None, restrict_to="", min_margin=0.0):
    """Concatenate frames of one role, slicing the requested radius shells.

    ``restrict_to`` keeps only centres that also survived in another cache. A
    longer integration loses ~10% of stars to the validity mask, and not at
    random, so comparing arc lengths on their own survivor sets confounds the
    integration length with a shifted sample. Restricting every cache to a
    common survivor set removes that.
    """
    def shell_index(available):
        """Slice indices for ``shells`` given the radii a cache actually stores."""
        out = [0]
        for s in shells:
            where = np.flatnonzero(np.isclose(available, s))
            if not len(where):
                raise ValueError(f"cache has radii {list(available)}, asked for shell {s}")
            b = 1 + NEIGHBOUR_COUNT * int(where[0])
            out.extend(range(b, b + NEIGHBOUR_COUNT))
        return np.asarray(out)

    idx = None
    curves, labels, dist, scene = [], [], [], []
    for path in sorted(Path(cache).glob("*.npz")):
        with np.load(path, allow_pickle=False) as d:
            if str(d["role"]) != role:
                continue
            key = str(d["key"])
            if scenes and key.split(":")[0] not in scenes:
                continue
            keep = slice(None)
            if min_margin > 0:
                # Keep only centres at least ``min_margin`` * h clear of the boundary, so
                # caches built with different neighbour radii can be scored on the same
                # spatial region -- separating star geometry from coreline coverage.
                lo, hi = _frame_bounds(str(d["key"]))
                m = float(d["h"]) * min_margin * 1.05
                c = d["centres"]
                keep = np.all((c >= lo + m) & (c <= hi - m), axis=1)
                if not keep.any():
                    continue
            if restrict_to:
                allowed = None
                for root in restrict_to.split(","):
                    other = Path(root) / path.name
                    if not other.exists():
                        allowed = set()
                        break
                    with np.load(other, allow_pickle=False) as d2:
                        here = {c.tobytes() for c in d2["centres"]}
                    allowed = here if allowed is None else (allowed & here)
                if not allowed:
                    continue
                sel = np.array([c.tobytes() in allowed for c in d["centres"]], dtype=bool)
                keep = sel if isinstance(keep, slice) else (keep & sel)
                if not keep.any():
                    continue
            if idx is None:
                idx = shell_index(d["radii"])
            curves.append(d["curves"][:, idx][keep])
            labels.append(d["labels"][keep])
            dist.append(d["dist"][keep])
            n_kept = len(d["labels"]) if isinstance(keep, slice) else int(np.sum(keep))
            scene.extend([key.split(":")[0]] * n_kept)
    if not curves:
        raise FileNotFoundError(f"no {role} frames in {cache} (scenes={scenes})")
    return (np.concatenate(curves), np.concatenate(labels).astype(np.int64),
            np.concatenate(dist), np.array(scene))


def macro_f1(pred, true):
    out = []
    for c in (0, 1):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))


@torch.no_grad()
def predict(model, head, sig, stats, clip, batch=4096):
    mu_device = next(model.parameters()).device
    model.eval(); head.eval()
    out = []
    for i in range(0, len(sig), batch):
        x = apply_norm_stats_ico(sig[i:i + batch].to(mu_device), stats, clip)
        mu, _ = model.encode(x)
        out.append(head(mu[:, :model.z_inv_dim]).argmax(-1).cpu())
    model.train(); head.train()
    return torch.cat(out).numpy()


class Projection(nn.Module):
    """Contrastive projection head.

    Applying the contrastive loss straight to ``z_inv`` forces the classifier's
    own feature space to be group-invariant, which can erase discriminative
    structure. Projecting first lets the encoder stay discriminative while the
    *projected* space carries the invariance -- the standard SimCLR arrangement.
    """

    def __init__(self, z_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, z):
        return self.net(z)


class Head(nn.Module):
    def __init__(self, z, hidden=128, dropout=0.15, kind="mlp"):
        super().__init__()
        self.net = nn.Linear(z, 2) if kind == "linear" else nn.Sequential(
            nn.Linear(z, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, z):
        return self.net(z)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/home/cheny1a/data/task6_icostar_1_1")
    p.add_argument("--output", default="outputs/exp_Task6_NewStar")
    p.add_argument("--label", default="run")
    p.add_argument("--shells", default="1", help="comma list from 1,2,4,8")
    p.add_argument("--scenes", default="")
    p.add_argument("--data-device", default="cuda", choices=["cuda", "cpu"],
                   help="where the signal tensors live; cpu streams batches to the GPU "
                        "for caches too large to hold in device memory")
    p.add_argument("--min-margin", type=float, default=0.0,
                   help="keep only centres this many h clear of the boundary, so caches "
                        "with different radii can be scored on the same spatial region")
    p.add_argument("--restrict-to", default="",
                   help="comma-separated caches; keep only centres that survived in ALL of "
                        "them, so arc lengths are compared on exactly the same stars")
    p.add_argument("--seed", type=int, default=7068)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--head", default="mlp", choices=["mlp", "linear"])
    p.add_argument("--z-inv", type=int, default=16)
    p.add_argument("--z-eq", type=int, default=12)
    p.add_argument("--clip-sigma", type=float, default=5.0)
    p.add_argument("--aug", default="ih", choices=["ih", "i", "so3", "so3snap", "both"])
    p.add_argument("--lambda-ssl", type=float, default=0.0)
    p.add_argument("--lambda-recon", type=float, default=0.0)
    p.add_argument("--zinv-var-weight", type=float, default=0.0)
    p.add_argument("--proj-dim", type=int, default=0,
                   help="project z_inv to this width before the contrastive loss (0 = apply it "
                        "directly to z_inv, as in earlier rounds)")
    p.add_argument("--aux-decay", action="store_true",
                   help="anneal the auxiliary weights to zero on the same cosine as the LR, so "
                        "they shape early training and leave the final fit unbiased")
    p.add_argument("--transductive", action="store_true",
                   help="let the SSL / reconstruction terms draw from the TEST stars too (and, "
                        "under --holdout-scene, from the held-out scene's train-role frames). "
                        "Labels are never used for those rows, only their geometry -- the standard "
                        "transductive / unsupervised-domain-adaptation setting.")
    p.add_argument("--holdout-scene", default="",
                   help="train on every other scene and test only on this one (leave-one-scene-out)")
    p.add_argument("--encoder", default="conv", choices=["conv", "pure"])
    p.add_argument("--schedule", default="constant", choices=["constant", "cosine"])
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--min-lr-scale", type=float, default=0.0)
    p.add_argument("--class-weight", action="store_true")
    p.add_argument("--pretrain-steps", type=int, default=0,
                   help="SSL/reconstruction-only steps on the full unlabelled pool before "
                        "the supervised phase (canonical two-stage protocol)")
    p.add_argument("--drop-aux-after-pretrain", action="store_true",
                   help="zero the SSL/reconstruction weights once pretraining ends, so stage 2 is "
                        "purely supervised (a true two-stage protocol rather than pretrain+joint)")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="after pretraining, train only the head (linear probe)")
    p.add_argument("--label-fraction", type=float, default=1.0,
                   help="fraction of training stars whose labels the head may use; "
                        "SSL and reconstruction still see every star")
    p.add_argument("--save-weights", default="")
    args = p.parse_args()

    shells = tuple(float(x) for x in args.shells.split(","))
    scenes = tuple(x for x in args.scenes.split(",") if x)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_scenes, test_scenes = scenes, scenes
    if args.holdout_scene:
        # Leave-one-scene-out: never see the held-out flow during training. This is a
        # domain-generalisation setting, where an invariance prior has the most to offer.
        every = sorted({p.name.rsplit("_", 1)[0] for p in Path(args.cache).glob("*.npz")})
        train_scenes = tuple(s for s in every if s != args.holdout_scene)
        test_scenes = (args.holdout_scene,)
        if args.holdout_scene not in every:
            raise ValueError(f"unknown scene {args.holdout_scene!r}; have {every}")
    tr_c, tr_y, _, _ = load(args.cache, "train", shells, train_scenes, args.restrict_to, args.min_margin)
    te_c, te_y, te_d, te_s = load(args.cache, "test", shells, test_scenes, args.restrict_to, args.min_margin)
    store = device if args.data_device == "cuda" else "cpu"
    tr_sig = torch.from_numpy(build_signal_ico(tr_c)).to(store)
    te_sig = torch.from_numpy(build_signal_ico(te_c)).to(store)
    del tr_c, te_c
    _, stats = normalize_signal_ico(tr_sig[:8192].to(device), args.clip_sigma)
    chan_w = channel_balance_weights_ico(
        apply_norm_stats_ico(tr_sig[:4096].to(device), stats, args.clip_sigma))

    # Auxiliary pool. The head only ever sees `labelled` rows of tr_sig; the SSL and
    # reconstruction terms may additionally see unlabelled target-domain geometry.
    aux_sig = tr_sig
    aux_note = "train only"
    if args.transductive:
        pools = [tr_sig, te_sig]
        if args.holdout_scene:
            extra_c, _, _, _ = load(args.cache, "train", shells, (args.holdout_scene,),
                                    args.restrict_to, args.min_margin)
            pools.append(torch.from_numpy(build_signal_ico(extra_c)).to(store))
            del extra_c
        aux_sig = torch.cat(pools, dim=0)
        aux_note = f"train + unlabelled target ({len(aux_sig) - len(tr_sig):,} extra rows)"
    print(f"  aux pool: {len(aux_sig):,} rows ({aux_note}); labelled rows: {len(tr_y):,}", flush=True)

    lines = tr_sig.shape[1]
    model = IcosahedralSirenVAE3D(steps=tr_sig.shape[2], z_inv_dim=args.z_inv,
                                  z_eq_dim=args.z_eq, encoder_kind=args.encoder,
                                  lines=lines).to(device)
    head = Head(args.z_inv, kind=args.head).to(device)
    proj = Projection(args.z_inv, args.proj_dim).to(device) if args.proj_dim > 0 else None
    matrices, permutation = group_tensors("ih" if args.aug != "i" else "i", device=device)

    params = list(model.parameters()) + list(head.parameters())
    if proj is not None:
        params += list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def lr_factor(step):
        if args.warmup_steps and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        if args.schedule == "cosine":
            t = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            return args.min_lr_scale + (1 - args.min_lr_scale) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    tr_y_t = torch.from_numpy(tr_y).to(device)
    cw = None
    if args.class_weight:
        counts = np.bincount(tr_y, minlength=2).astype(np.float64)
        cw = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)

    if args.label_fraction < 1.0:
        rs = np.random.default_rng(args.seed)
        keep = rs.permutation(len(tr_y))[:max(int(round(len(tr_y) * args.label_fraction)), 2)]
        # keep at least one of each class so cross-entropy stays defined
        for c in (0, 1):
            if not (tr_y[keep] == c).any():
                keep = np.append(keep, np.flatnonzero(tr_y == c)[0])
        labelled = torch.from_numpy(np.sort(keep)).to(device)
    else:
        labelled = torch.arange(len(tr_y), device=device)

    gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    curve, best = [], {"f1": -1.0}
    t0 = time.time()

    if args.pretrain_steps > 0:
        # Stage 1: no labels at all -- SSL and/or reconstruction over every training star.
        pre = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for step in range(args.pretrain_steps):
            j = torch.randint(0, len(aux_sig), (args.batch,), device=device, generator=gen)
            raw = aux_sig[j.to(aux_sig.device)].to(device)
            x = apply_norm_stats_ico(raw, stats, args.clip_sigma)
            loss = x.new_zeros(())
            if args.lambda_ssl > 0 or args.zinv_var_weight > 0:
                v = views_of(raw, stats, args.clip_sigma, matrices, permutation, args.aug, gen)
                zs = [model.encode(a)[0][:, :model.z_inv_dim] for a in v]
            if proj is not None:
                zs = [proj(z) for z in zs]
                loss = loss + args.lambda_ssl * multi_view_contrastive_loss(zs)
                if args.zinv_var_weight > 0:
                    var, cov = zinv_variance_covariance(zs)
                    loss = loss + args.zinv_var_weight * aux_w * (var + cov)
            if args.lambda_recon > 0:
                mu, logvar = model.encode(x)
                z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
                rl, _, _, _ = vae_ico_loss(model.decode(z), x, mu, logvar, mu.new_zeros(()),
                                           lambda_recon=1.0, lambda_contrast=0.0,
                                           channel_weights=chan_w)
                loss = loss + args.lambda_recon * aux_w * rl
            if not loss.requires_grad:
                raise ValueError("--pretrain-steps needs --lambda-ssl or --lambda-recon")
            pre.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); pre.step()
        print(f"  pretrained {args.pretrain_steps} steps", flush=True)
        if args.drop_aux_after_pretrain:
            args.lambda_ssl = args.lambda_recon = args.zinv_var_weight = 0.0
        if args.freeze_encoder:
            for q in model.encoder.parameters():
                q.requires_grad_(False)
            opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    for step in range(args.steps):
        i = labelled[torch.randint(0, len(labelled), (args.batch,), device=device, generator=gen)]
        j = torch.randint(0, len(aux_sig), (args.batch,), device=device, generator=gen)
        raw = tr_sig[i.to(tr_sig.device)].to(device)
        aux_w = lr_factor(step) if args.aux_decay else 1.0
        x = apply_norm_stats_ico(raw, stats, args.clip_sigma)
        mu, logvar = model.encode(x)
        logits = head(mu[:, :model.z_inv_dim])
        loss = F.cross_entropy(logits, tr_y_t[i], weight=cw)
        parts = {"ce": float(loss)}

        if args.lambda_ssl > 0 or args.zinv_var_weight > 0:
            v = views_of(aux_sig[j.to(aux_sig.device)].to(device), stats, args.clip_sigma,
                         matrices, permutation, args.aug, gen)
            zs = [model.encode(a)[0][:, :model.z_inv_dim] for a in v]
            if proj is not None:
                zs = [proj(z) for z in zs]
            con = multi_view_contrastive_loss(zs)
            loss = loss + args.lambda_ssl * aux_w * con
            parts["ssl"] = float(con)
            if args.zinv_var_weight > 0:
                var, cov = zinv_variance_covariance(zs)
                loss = loss + args.zinv_var_weight * aux_w * (var + cov)
        if args.lambda_recon > 0:
            # Reconstruction is label-free, so under --transductive it should see the
            # unlabelled pool too rather than only the labelled batch.
            if args.transductive:
                xr = apply_norm_stats_ico(aux_sig[j.to(aux_sig.device)].to(device),
                                          stats, args.clip_sigma)
                mur, logvarr = model.encode(xr)
            else:
                xr, mur, logvarr = x, mu, logvar
            z = mur + torch.randn_like(mur) * (0.5 * logvarr).exp()
            rec = model.decode(z)
            rl, _, _, _ = vae_ico_loss(rec, xr, mur, logvarr, mur.new_zeros(()),
                                       lambda_recon=1.0, lambda_contrast=0.0,
                                       channel_weights=chan_w)
            loss = loss + args.lambda_recon * aux_w * rl
            parts["recon"] = float(rl)

        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            pred = predict(model, head, te_sig, stats, args.clip_sigma)
            f1 = macro_f1(pred, te_y)
            per = {s: macro_f1(pred[te_s == s], te_y[te_s == s]) for s in np.unique(te_s)}
            curve.append({"step": step + 1, "f1": f1, "lr": sched.get_last_lr()[0],
                          "per_scene": per, **parts})
            if f1 > best["f1"]:
                best = {"f1": f1, "step": step + 1, "per_scene": per}
                if args.save_weights:
                    Path(args.save_weights).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": model.state_dict(), "head": head.state_dict(),
                                "args": vars(args), "stats": stats, "f1": f1},
                               args.save_weights)
            print(f"  step {step+1:5d} f1 {f1:.4f} best {best['f1']:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rec = dict(label=args.label, labelled_n=int(len(labelled)),
               train_scenes=list(train_scenes), test_scenes=list(test_scenes),
               aux_pool_n=int(len(aux_sig)), transductive=bool(args.transductive), args=vars(args), shells=list(shells), lines=lines,
               train_n=int(len(tr_y)), test_n=int(len(te_y)),
               train_pos=float(tr_y.mean()), test_pos=float(te_y.mean()),
               final_f1=curve[-1]["f1"], best=best, curve=curve,
               params=model.parameter_counts(), seconds=time.time() - t0)
    (out / f"{args.label}.json").write_text(json.dumps(rec, indent=2))
    print(f"[done] {args.label} best {best['f1']:.4f} final {curve[-1]['f1']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
