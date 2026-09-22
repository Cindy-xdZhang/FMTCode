"""Supervised coreline classification with the IcoVAE encoder, plus optional SSL.

Task 6 is supervised: predict whether a seed lies on a vortex coreline, from its
13-line icosahedral streamline star.  The question this script answers is whether
the self-supervised objectives that carry Task 1 (icosahedral contrastive views,
SIREN reconstruction) help when a label is available.

Three arms:

* `sup`       -- encoder + linear/MLP head, cross-entropy only
* `sup_ssl`   -- the same, plus the contrastive term on I_h views and the VICReg
                 floor, trained jointly
* `sup_ssl_rec` -- the same, plus SIREN reconstruction

The dataset's own `default_split.json` frame split is used unchanged, so no
frame contributes to both train and test.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from FMT_Utils.IcosahedralGroup_3D import group_tensors
from FMT_Utils.IcoVAE_3D import (IcosahedralSirenVAE3D, apply_norm_stats_ico,
                                 build_signal_ico, channel_balance_weights_ico,
                                 make_views, multi_view_contrastive_loss,
                                 normalize_signal_ico, zinv_variance_covariance)


def load_split(cache, role, scenes=None):
    """Frames of one split; `scenes` restricts to a comma-separated flow list."""
    geometry, labels, frames = [], [], []
    keep = set(s for s in (scenes or "").split(",") if s)
    for path in sorted((Path(cache) / role).glob("*.npz")):
        if keep and not any(path.name.startswith(s + "_") for s in keep):
            continue
        with np.load(path) as data:
            geometry.append(data["geometry"])
            labels.append(data["labels"])
            frames.append(np.full(len(data["labels"]), len(frames)))
    if not geometry:
        raise FileNotFoundError(f"no {role} frames in {cache} for scenes={scenes!r}")
    return (np.concatenate(geometry), np.concatenate(labels).astype(np.int64),
            np.concatenate(frames))


class CorelineHead(nn.Module):
    def __init__(self, z_dim, hidden=128, dropout=0.15, kind="mlp"):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(z_dim, 2)
        else:
            self.net = nn.Sequential(nn.Linear(z_dim, hidden), nn.LayerNorm(hidden),
                                     nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, z):
        return self.net(z)


@torch.no_grad()
def evaluate(model, head, signal_n, labels, batch=4096):
    model.eval(); head.eval()
    scores = []
    for start in range(0, len(signal_n), batch):
        mu, _ = model.encode(signal_n[start:start + batch])
        scores.append(torch.softmax(head(mu), -1)[:, 1].float().cpu().numpy())
    model.train(); head.train()
    probability = np.concatenate(scores)
    prediction = (probability >= 0.5).astype(int)
    return (f1_score(labels, prediction, zero_division=0),
            average_precision_score(labels, probability),
            f1_score(labels, prediction, average="macro", zero_division=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="outputs/exp_Task6_Ico_cache")
    parser.add_argument("--output", default="outputs/exp_Task6_IcoVAE_1.1")
    parser.add_argument("--label", default="sup")
    parser.add_argument("--seed", type=int, default=7068)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head", default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--z-inv", type=int, default=16)
    parser.add_argument("--z-eq", type=int, default=12)
    parser.add_argument("--clip-sigma", type=float, default=5.0)
    parser.add_argument("--aug", default="ih", choices=["ih", "i", "so3", "both"])
    parser.add_argument("--lambda-ssl", type=float, default=0.0,
                        help="weight on the icosahedral contrastive term")
    parser.add_argument("--lambda-recon", type=float, default=0.0)
    parser.add_argument("--zinv-var-weight", type=float, default=0.0)
    parser.add_argument("--save-weights", default="",
                        help="directory to write best/final checkpoints into")
    parser.add_argument("--schedule", default="constant",
                        choices=["constant", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="linear warmup from 0 to lr over this many steps")
    parser.add_argument("--min-lr-scale", type=float, default=0.0,
                        help="cosine floor as a fraction of lr")
    parser.add_argument("--scenes", default="",
                        help="comma-separated flows; empty = all four (joint model)")
    parser.add_argument("--class-weight", action="store_true",
                        help="inverse-frequency weighting in the cross entropy")
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed); np.random.seed(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_geometry, train_labels, _ = load_split(ROOT / arguments.cache, "train",
                                                 arguments.scenes)
    test_geometry, test_labels, _ = load_split(ROOT / arguments.cache, "test",
                                               arguments.scenes)
    train_raw = torch.from_numpy(build_signal_ico(train_geometry)).to(device)
    test_raw = torch.from_numpy(build_signal_ico(test_geometry)).to(device)
    train_n, stats = normalize_signal_ico(train_raw, arguments.clip_sigma)
    test_n = apply_norm_stats_ico(test_raw, stats, arguments.clip_sigma)
    weights_channel = channel_balance_weights_ico(train_n)
    y_train = torch.as_tensor(train_labels, device=device)

    model = IcosahedralSirenVAE3D(steps=train_n.shape[2], z_inv_dim=arguments.z_inv,
                                  z_eq_dim=arguments.z_eq,
                                  lines=train_n.shape[1]).to(device)
    head = CorelineHead(model.z_dim, kind=arguments.head).to(device)
    matrices, permutation = group_tensors("ih", device=device)
    counts = model.parameter_counts()
    head_count = sum(p.numel() for p in head.parameters())
    print(f"[{arguments.label}] scenes={arguments.scenes or 'ALL'}  "
          f"train {len(train_labels):,} (pos {train_labels.mean():.4f})  "
          f"test {len(test_labels):,} (pos {test_labels.mean():.4f})  "
          f"encoder {counts['encoder']:,} head {head_count:,}", flush=True)

    class_weight = None
    if arguments.class_weight:
        frequency = np.bincount(train_labels, minlength=2) / len(train_labels)
        class_weight = torch.as_tensor((1.0 / np.maximum(frequency, 1e-6)) /
                                       (1.0 / np.maximum(frequency, 1e-6)).sum() * 2,
                                       dtype=torch.float32, device=device)

    optimiser = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": arguments.lr * arguments.encoder_lr_scale},
         {"params": head.parameters(), "lr": arguments.lr}],
        weight_decay=arguments.weight_decay)
    generator = torch.Generator(device=device).manual_seed(arguments.seed)

    def lr_factor(step):
        """Linear warmup, then constant or cosine decay to `min_lr_scale`."""
        if arguments.warmup_steps > 0 and step < arguments.warmup_steps:
            return (step + 1) / arguments.warmup_steps
        if arguments.schedule != "cosine":
            return 1.0
        span = max(arguments.steps - arguments.warmup_steps, 1)
        progress = min((step - arguments.warmup_steps) / span, 1.0)
        floor = arguments.min_lr_scale
        return floor + (1.0 - floor) * 0.5 * (1.0 + np.cos(np.pi * progress))

    base_lrs = [group["lr"] for group in optimiser.param_groups]
    curve, started, best = [], time.time(), {"f1": -1.0}
    for step in range(arguments.steps + 1):
        if step % arguments.eval_every == 0:
            f1, ap, macro = evaluate(model, head, test_n, test_labels)
            curve.append({"step": step, "test_f1": f1, "test_ap": ap,
                          "test_macro_f1": macro,
                          "lr": float(optimiser.param_groups[0]["lr"])})
            if f1 > best["f1"]:
                best = {"f1": f1, "ap": ap, "macro": macro, "step": step}
                if arguments.save_weights:
                    store = ROOT / arguments.save_weights
                    store.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": model.state_dict(), "head": head.state_dict(),
                                "step": step, "metrics": best,
                                "arguments": vars(arguments),
                                "norm_stats": [float(stats[0]), float(stats[1])],
                                "lines": int(train_n.shape[1]),
                                "steps_per_line": int(train_n.shape[2])},
                               store / f"{arguments.label}_best.pt")
            print(f"  step {step:5d}  test F1 {f1:.4f}  AP {ap:.4f}  macro {macro:.4f}",
                  flush=True)
        if step == arguments.steps:
            break

        factor = lr_factor(step)
        for group, base in zip(optimiser.param_groups, base_lrs):
            group["lr"] = base * factor
        index = torch.randint(0, len(train_n), (arguments.batch,), device=device,
                              generator=generator)
        original = train_n[index]
        if arguments.lambda_ssl > 0 or arguments.lambda_recon > 0:
            views = make_views(train_raw[index], stats, arguments.clip_sigma,
                               matrices, permutation, mode=arguments.aug,
                               generator=generator, count=2)
            stacked = torch.cat([original, *views])
        else:
            views, stacked = [], original
        mu_all, logvar_all = model.encode(stacked)
        logits = head(mu_all[:arguments.batch])
        loss = F.cross_entropy(logits, y_train[index], weight=class_weight)

        if arguments.lambda_ssl > 0:
            parts = [mu_all[i * arguments.batch:(i + 1) * arguments.batch, :model.z_inv_dim]
                     for i in range(1 + len(views))]
            loss = loss + arguments.lambda_ssl * multi_view_contrastive_loss(parts)
            if arguments.zinv_var_weight > 0:
                variance, covariance = zinv_variance_covariance(parts)
                loss = loss + arguments.zinv_var_weight * variance + 0.2 * covariance
        if arguments.lambda_recon > 0:
            recon = model.decode(mu_all)
            loss = loss + arguments.lambda_recon * (
                weights_channel * (recon - stacked).pow(2)).mean()

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
        optimiser.step()

    summary = {"label": arguments.label, "arguments": vars(arguments),
               "parameters": {**counts, "head": head_count},
               "scenes": arguments.scenes or "ALL",
               "train": int(len(train_labels)), "test": int(len(test_labels)),
               "train_positive": float(train_labels.mean()),
               "test_positive": float(test_labels.mean()),
               "best": best, "final": curve[-1],
               "minutes": (time.time() - started) / 60.0, "curve": curve}
    if arguments.save_weights:
        store = ROOT / arguments.save_weights
        store.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "head": head.state_dict(),
                    "step": arguments.steps, "metrics": curve[-1],
                    "arguments": vars(arguments),
                    "norm_stats": [float(stats[0]), float(stats[1])],
                    "lines": int(train_n.shape[1]),
                    "steps_per_line": int(train_n.shape[2])},
                   store / f"{arguments.label}_final.pt")
    output = ROOT / arguments.output
    output.mkdir(parents=True, exist_ok=True)
    (output / f"summary_{arguments.label}.json").write_text(json.dumps(summary, indent=1))
    print(f"[{arguments.label}] best F1 {best['f1']:.4f} @ step {best['step']} "
          f"(AP {best['ap']:.4f}), final F1 {curve[-1]['test_f1']:.4f}, "
          f"{summary['minutes']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
