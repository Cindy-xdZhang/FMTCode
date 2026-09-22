"""Icosahedral SIREN-VAE clustering on 3D pathline primitives — unsupervised.

No split.  Every timeslice of a dataset is training data, and the IVD reference
is a **metric only**: it never enters a loss, a checkpoint choice, a read-out
rule or a hyper-parameter.  The whole F1 curve is recorded at every evaluation
for every read-out rule, so the question "how should a good representation be
picked for clustering?" is answered from the record rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from FMT_Utils.ClusterReadout_3D import RULES, cluster_readouts, rule_correlations
from FMT_Utils.IcosahedralGroup_3D import group_tensors
from FMT_Utils.IcoVAE_3D import (IcosahedralSirenVAE3D, apply_norm_stats_ico,
                                 build_signal_ico, channel_balance_weights_ico,
                                 make_views, multi_view_contrastive_loss,
                                 normalize_signal_ico, vae_ico_loss,
                                 zinv_variance_covariance)


def load_dataset(cache, dataset):
    folder = Path(cache) / dataset
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no cache slices in {folder}")
    geometry, reference, slices = [], [], []
    for index, path in enumerate(files):
        with np.load(path) as data:
            geometry.append(data["geometry"])
            reference.append(data["reference"])
            slices.append(np.full(len(data["reference"]), index))
    return (np.concatenate(geometry), np.concatenate(reference).astype(int),
            np.concatenate(slices), [p.name for p in files])


@torch.no_grad()
def encode_all(model, signal_n, batch=4096):
    model.eval()
    parts = []
    for start in range(0, len(signal_n), batch):
        mu, _ = model.encode(signal_n[start:start + batch])
        parts.append(mu[:, :model.z_inv_dim].float().cpu().numpy())
    model.train()
    return np.concatenate(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="outputs/exp_Task1_IcoVAE_cache")
    parser.add_argument("--dataset", default="halfcylinderRe160_eth")
    parser.add_argument("--output", default="outputs/exp_Task1_IcoVAE_1.1")
    parser.add_argument("--label", default="base")
    parser.add_argument("--seed", type=int, default=7068)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aug", default="ih", choices=["ih", "i", "so3", "both"])
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--encoder", default="conv", choices=["conv", "pure"])
    parser.add_argument("--z-inv", type=int, default=16)
    parser.add_argument("--z-eq", type=int, default=12)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-contrast", type=float, default=15.0)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--zinv-var-weight", type=float, default=5.0)
    parser.add_argument("--zinv-cov-weight", type=float, default=0.2)
    parser.add_argument("--zinv-target-std", type=float, default=0.5)
    parser.add_argument("--clip-sigma", type=float, default=5.0)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--clusterer", default="gmm_tied",
                        help="kmeans | gmm_tied | gmm_full | gmm_diag | spectral | agglomerative_*")
    parser.add_argument("--save-weights", default="",
                        help="directory to write the selected/final checkpoints into")
    parser.add_argument("--save-latents", default="",
                        help="directory to store z_inv at every evaluation (float16)")
    parser.add_argument("--cosine", action="store_true", help="cosine-anneal the lr")
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed); np.random.seed(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    geometry, reference, slice_id, files = load_dataset(ROOT / arguments.cache, arguments.dataset)
    signal_raw = torch.from_numpy(build_signal_ico(geometry)).to(device)
    signal_n, stats = normalize_signal_ico(signal_raw, arguments.clip_sigma)
    weights = channel_balance_weights_ico(signal_n)
    matrices, permutation = group_tensors("ih" if arguments.aug in ("ih", "both") else "i",
                                          device=device)
    model = IcosahedralSirenVAE3D(steps=signal_n.shape[2], z_inv_dim=arguments.z_inv,
                                  z_eq_dim=arguments.z_eq,
                                  encoder_kind=arguments.encoder).to(device)
    counts = model.parameter_counts()
    print(f"[{arguments.label}] {arguments.dataset}: {len(geometry):,} primitives from "
          f"{len(files)} slices, positives {reference.mean():.4f}, "
          f"params {counts['total']:,}", flush=True)

    optimiser = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": arguments.lr * arguments.encoder_lr_scale},
         {"params": model.decoder.parameters(), "lr": arguments.lr}],
        weight_decay=arguments.weight_decay)
    schedule = (torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=arguments.steps)
                if arguments.cosine else None)
    generator = torch.Generator(device=device).manual_seed(arguments.seed)

    rows = len(signal_n)
    curve, started = [], time.time()
    for step in range(arguments.steps + 1):
        if step % arguments.eval_every == 0:
            latent = encode_all(model, signal_n)
            if arguments.save_latents:
                store = ROOT / arguments.save_latents / arguments.dataset
                store.mkdir(parents=True, exist_ok=True)
                np.save(store / f"{arguments.label}_step{step:06d}.npy",
                        latent.astype(np.float16))
            scores, diagnostics = cluster_readouts(latent, reference,
                                                   restarts=arguments.restarts,
                                                   seed=arguments.seed,
                                                   clusterer=arguments.clusterer)
            entry = {"step": step, "rules": scores,
                     "correlations": rule_correlations(diagnostics),
                     "best_silhouette": max((d["silhouette"] for d in diagnostics), default=0.0),
                     "min_davies_bouldin": min((d["davies_bouldin"] for d in diagnostics),
                                               default=0.0),
                     "minority_fraction": float(np.median([d["minority_fraction"]
                                                           for d in diagnostics])) if diagnostics else 0.0,
                     "latent_std": float(latent.std())}
            curve.append(entry)
            if arguments.save_weights and (
                    entry["min_davies_bouldin"] <= min(c["min_davies_bouldin"] for c in curve)):
                store = ROOT / arguments.save_weights / arguments.dataset
                store.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "step": step, "entry": entry,
                            "arguments": vars(arguments),
                            "norm_stats": [float(stats[0]), float(stats[1])],
                            "lines": int(signal_n.shape[1]),
                            "steps_per_line": int(signal_n.shape[2])},
                           store / f"{arguments.label}_selected.pt")
            print(f"  step {step:6d}  " + "  ".join(
                f"{rule[:4]}={scores[rule]:.4f}" for rule in RULES)
                + f"  sil={entry['best_silhouette']:.3f}", flush=True)
        if step == arguments.steps:
            break

        index = torch.randint(0, rows, (arguments.batch,), device=device, generator=generator)
        original = signal_n[index]
        views = make_views(signal_raw[index], stats, arguments.clip_sigma,
                           matrices, permutation, mode=arguments.aug,
                           generator=generator, count=arguments.views)
        stacked = torch.cat([original, *views])
        mu_all, logvar_all = model.encode(stacked)
        parts = [mu_all[i * arguments.batch:(i + 1) * arguments.batch, :model.z_inv_dim]
                 for i in range(1 + len(views))]
        contrastive = multi_view_contrastive_loss(parts)
        recon = model.decode(mu_all) if arguments.lambda_recon > 0 else None
        loss, recon_value, kl_value, contrast_value = vae_ico_loss(
            recon, stacked, mu_all, logvar_all, contrastive, beta=arguments.beta,
            lambda_recon=arguments.lambda_recon, lambda_contrast=arguments.lambda_contrast,
            channel_weights=weights)
        if arguments.zinv_var_weight > 0 or arguments.zinv_cov_weight > 0:
            variance, covariance = zinv_variance_covariance(
                parts, target_std=arguments.zinv_target_std)
            loss = loss + arguments.zinv_var_weight * variance + arguments.zinv_cov_weight * covariance
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if schedule is not None:
            schedule.step()

    summary = {"label": arguments.label, "dataset": arguments.dataset,
               "arguments": vars(arguments), "parameters": counts,
               "primitives": int(len(geometry)), "slices": files,
               "positive_fraction": float(reference.mean()),
               "minutes": (time.time() - started) / 60.0, "curve": curve}
    for rule in RULES:
        values = [c["rules"][rule] for c in curve]
        summary[f"{rule}_selected_by_db"] = values[int(np.argmin(
            [c["min_davies_bouldin"] for c in curve]))]
        summary[f"{rule}_best"] = max(values)
        summary[f"{rule}_end"] = values[-1]
        summary[f"{rule}_mean"] = float(np.mean(values))
    if arguments.save_latents:
        store = ROOT / arguments.save_latents / arguments.dataset
        store.mkdir(parents=True, exist_ok=True)
        np.save(store / "reference.npy", reference.astype(np.int8))
    if arguments.save_weights:
        store = ROOT / arguments.save_weights / arguments.dataset
        store.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "step": arguments.steps,
                    "entry": curve[-1], "arguments": vars(arguments),
                    "norm_stats": [float(stats[0]), float(stats[1])],
                    "lines": int(signal_n.shape[1]),
                    "steps_per_line": int(signal_n.shape[2])},
                   store / f"{arguments.label}_final.pt")
    output = ROOT / arguments.output
    output.mkdir(parents=True, exist_ok=True)
    (output / f"summary_{arguments.label}_{arguments.dataset}.json").write_text(
        json.dumps(summary, indent=1))
    print(f"[{arguments.label}] {arguments.dataset} done in {summary['minutes']:.1f} min  "
          + "  ".join(f"{r}: mean {summary[f'{r}_mean']:.4f} best {summary[f'{r}_best']:.4f}"
                      for r in ("inertia", "silhouette", "oracle")), flush=True)


if __name__ == "__main__":
    main()
