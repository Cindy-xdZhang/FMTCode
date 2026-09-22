"""Two-stage Task4-c: self-supervised octahedral VAE, then a frozen-encoder probe.

Stage 1 trains the Task1 SIREN-VAE on the 7-line octahedral primitives with **no
label and no prediction head**.  The objective is the Task1 one: channel-balanced
reconstruction, KL, a positive-pairs contrastive term pulling `z_inv` together
across augmented views, and VICReg to stop that term collapsing the code.  Early
stopping uses Davies-Bouldin on `z_inv`, which needs no labels.

Stage 2 freezes the encoder, encodes every split once, and trains only the
prediction head (`mlp` or `linear`) on the frozen latents.

Augmentation (`--group`):

* `oh` / `o` -- exact lattice action: rotate the components **and** permute the
  six neighbour channels, because an octahedral element maps the seeding lattice
  onto itself.  Verified bitwise in `tests/test_task4c_octahedral_primitives.py`.
* `so3` -- a uniformly random rotation applied to the coordinates with the
  channel order left alone.  A general rotation sends the lattice off itself, so
  no permutation is defined; this is still a valid "same star seen from a rotated
  frame" view and teaches full SO(3) invariance of the shape.
* `none` -- augmentation off.

**The Davies-Bouldin guard.**  Task1 showed the failure mode this criterion has:
k-means can converge on a split that puts essentially everything in one cluster,
which scores a perfectly good Davies-Bouldin while being useless -- it showed up
there as predicted-positive fraction 1.0000 and ARI exactly 0.0000, with F1
pinned at the degenerate `2p/(1+p)`.  Two lessons are carried over here: the
latent is clustered L2-normalised and **never** standardised, and any checkpoint
whose split leaves a cluster below `--min-cluster-fraction` is rejected outright
rather than scored.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, davies_bouldin_score, f1_score
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.OctahedralGroup_3D import (  # noqa: E402
    apply_group, apply_norm_stats_oh, channel_balance_weights_oh, group_tensors,
    normalize_signal_oh,
)
from FMT_Utils.SirenVAE_3D import (  # noqa: E402
    OctahedralSirenVAE3D, l2_normalize, multi_view_contrastive_loss, reconstruction_mse,
    vae_oh_loss, zinv_variance_covariance,
)
from FMT_Utils.Task4C_FPSAugmentSearch_1_1 import mlp  # noqa: E402
from FMT_Utils.Task4C_OctVAE_2_1 import build_primitives, octahedral_selection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FLOWS = ("channel", "tbl")


def load_split(cache, split, flows_wanted=FLOWS):
    """Octahedral primitives, labels and flow ids for the usable bundles only.

    `flows_wanted` restricts to a single flow.  Channel and TBL are different
    simulations on incomparable physical scales (channel x in [0, 3.13], TBL x in
    [271, 352]), so pooling them puts a large nuisance axis into the latent: with
    both flows together the unsupervised 2-cluster split tracked flow identity
    (ARI .31) far more strongly than the Hairpin label (ARI .03), which is what
    the Davies-Bouldin criterion then optimised.  Training one flow at a time
    removes that axis and lets the train-only normalisation match the flow.
    """
    signals, labels, flows, kept, total = [], [], [], 0, 0
    for index, flow in enumerate(flows_wanted):
        folder = Path(cache) / flow / split
        geometry = np.load(folder / "geometry.npy")
        seeds = np.load(folder / "seeds.npy")
        with np.load(folder / "metadata.npz") as data:
            meta = {key: data[key] for key in data.files}
        select, usable = octahedral_selection(seeds, meta)
        signal, rows = build_primitives(geometry, seeds, meta, select, usable)
        signals.append(signal)
        labels.append(np.asarray(meta["labels"])[rows].astype(np.int64))
        flows.append(np.full(len(rows), index))
        kept += int(usable.sum()); total += int(len(usable))
    return (torch.cat(signals), np.concatenate(labels), np.concatenate(flows), kept, total)


def random_so3(count, generator, device):
    """Uniform rotations via QR of a Gaussian matrix, determinant fixed to +1."""
    matrix = torch.randn(count, 3, 3, device=device, generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(matrix)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    flip = torch.where(torch.det(q) < 0, -1.0, 1.0)
    q[:, :, 0] = q[:, :, 0] * flip.unsqueeze(-1)
    return q


def augment(signal, group, matrices, permutation, generator):
    if group == "so3":
        rotation = random_so3(len(signal), generator, signal.device)
        return torch.einsum("bij,bktj->bkti", rotation, signal)
    element = torch.randint(1, len(matrices), (len(signal),), device=signal.device,
                            generator=generator)
    return apply_group(signal, element, matrices, permutation)


def guarded_davies_bouldin(train_z, eval_z, seed, n_init, minimum_fraction):
    """Label-free score, with the Task1 degenerate-split guard."""
    train = l2_normalize(train_z); evaluate = l2_normalize(eval_z)   # never standardised
    model = KMeans(n_clusters=2, random_state=int(seed), n_init=int(n_init)).fit(train)
    labels = model.predict(evaluate)
    sizes = np.bincount(labels, minlength=2)
    fraction = float(sizes.min() / max(sizes.sum(), 1))
    if len(np.unique(labels)) < 2 or fraction < minimum_fraction:
        return float("-inf"), fraction          # degenerate: never selectable
    return -float(davies_bouldin_score(evaluate, labels)), fraction


@torch.no_grad()
def encode_all(model, signal, stats, clip, batch=4096):
    model.eval()
    out = []
    for start in range(0, len(signal), batch):
        chunk = apply_norm_stats_oh(signal[start:start + batch], stats, clip)
        out.append(model.encode(chunk)[0].float().cpu())
    model.train()
    return torch.cat(out)


def train_head(latents, labels, kind, z_dim, seed, device, epochs, lr, batch, dropout):
    torch.manual_seed(seed)
    if kind == "mlp":
        head = nn.Sequential(mlp(z_dim, 256, dropout), mlp(256, 128, dropout), nn.Linear(128, 2))
    else:
        head = nn.Linear(z_dim, 2)
    head = head.to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    train_x = latents["train"].to(device); train_y = torch.as_tensor(labels["train"], device=device)
    best = {"f1": -1.0, "ap": -1.0, "epoch": -1, "state": None}
    generator = torch.Generator(device=device).manual_seed(seed)
    for epoch in range(epochs):
        head.train()
        order = torch.randperm(len(train_x), device=device, generator=generator)
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            loss = F.cross_entropy(head(train_x[index]), train_y[index])
            optimiser.zero_grad(set_to_none=True); loss.backward(); optimiser.step()
        head.eval()
        with torch.no_grad():
            probability = torch.softmax(head(latents["validation"].to(device)), -1)[:, 1].cpu().numpy()
        truth = labels["validation"]
        metrics = (float(f1_score(truth, probability >= .5, zero_division=0)),
                   float(average_precision_score(truth, probability)))
        if metrics > (best["f1"], best["ap"]):
            best = {"f1": metrics[0], "ap": metrics[1], "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in head.state_dict().items()}}
    head.load_state_dict(best["state"]); head.eval()
    with torch.no_grad():
        test_probability = torch.softmax(head(latents["test"].to(device)), -1)[:, 1].cpu().numpy()
    return head, best, test_probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="outputs/exp_Task4C_RelaxedHead_1.0/physical")
    parser.add_argument("--seeds", default="96721")
    parser.add_argument("--group", default="oh", choices=["oh", "o", "so3", "none"])
    parser.add_argument("--heads", default="mlp,linear")
    parser.add_argument("--z-inv", type=int, default=144)
    parser.add_argument("--z-eq", type=int, default=112)
    parser.add_argument("--ssl-steps", type=int, default=3000)
    parser.add_argument("--ssl-batch", type=int, default=1024)
    parser.add_argument("--ssl-lr", type=float, default=5e-5)
    parser.add_argument("--lr-warmup", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--selection-warmup", type=int, default=200)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.02)
    parser.add_argument("--lambda-contrast", type=float, default=15.0)
    parser.add_argument("--zinv-var-weight", type=float, default=5.0)
    parser.add_argument("--zinv-cov-weight", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--clip-sigma", type=float, default=5.0)
    parser.add_argument("--head-epochs", type=int, default=200)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--head-batch", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--encoder", default="conv", choices=["conv", "pure"],
                        help="'pure' drops the conv front end so all mixing is attention")
    parser.add_argument("--output", default="outputs/exp_Task4C_TwoStage_1.1")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--also-final", action="store_true",
                        help="additionally probe the last stage-1 checkpoint, for a paired "
                             "comparison against the Davies-Bouldin selected one")
    parser.add_argument("--flow", default=None, choices=list(FLOWS),
                        help="train on a single flow; channel and TBL are separate simulations")
    parser.add_argument("--db-per-flow", action="store_true",
                        help="score Davies-Bouldin within each flow and average, removing the "
                             "flow nuisance that dominates the unsupervised split")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = ROOT / arguments.cache
    wanted = (arguments.flow,) if arguments.flow else FLOWS
    data, labels, flows = {}, {}, {}
    for split in ("train", "validation", "test"):
        signal, label, flow, kept, total = load_split(cache, split, wanted)
        data[split] = signal.to(device); labels[split] = label; flows[split] = flow
        print(f"{split:11s} usable {kept:,}/{total:,} ({kept/total:.3f})  "
              f"positives {label.mean():.4f}", flush=True)

    output = ROOT / arguments.output
    if arguments.tag:
        output = output.parent / f"{output.name}_{arguments.tag}"
    output.mkdir(parents=True, exist_ok=True)
    matrices, permutation = group_tensors("oh" if arguments.group == "so3" else
                                          (arguments.group if arguments.group != "none" else "oh"),
                                          device=device)
    results = []

    for seed in [int(s) for s in arguments.seeds.split(",")]:
        torch.manual_seed(seed); np.random.seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        started = time.time()

        train_n, stats = normalize_signal_oh(data["train"], arguments.clip_sigma)
        weights = channel_balance_weights_oh(train_n)
        model = OctahedralSirenVAE3D(steps=data["train"].shape[2], z_inv_dim=arguments.z_inv,
                                     z_eq_dim=arguments.z_eq,
                                     encoder_kind=arguments.encoder).to(device)
        optimiser = torch.optim.AdamW(
            [{"params": model.encoder.parameters(), "lr": arguments.ssl_lr * 0.1},
             {"params": model.decoder.parameters(), "lr": arguments.ssl_lr}], weight_decay=1e-4)

        def scale(step):
            if arguments.lr_warmup > 0 and step < arguments.lr_warmup:
                return (step + 1) / arguments.lr_warmup
            progress = (step - arguments.lr_warmup) / max(arguments.ssl_steps - arguments.lr_warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, scale)
        selection = torch.randperm(len(train_n), device=device,
                                   generator=generator)[:8192]
        best = {"score": float("-inf"), "step": -1, "state": None, "fraction": 0.0}
        history, step, cursor = [], 0, 0
        order = torch.randperm(len(train_n), device=device, generator=generator)

        while step < arguments.ssl_steps:
            if cursor + arguments.ssl_batch > len(order):
                order = torch.randperm(len(train_n), device=device, generator=generator); cursor = 0
            index = order[cursor:cursor + arguments.ssl_batch]; cursor += arguments.ssl_batch
            original = train_n[index]
            views = [original]
            if arguments.group != "none":
                for _ in range(2):
                    moved = augment(data["train"][index], arguments.group, matrices,
                                    permutation, generator)
                    views.append(apply_norm_stats_oh(moved, stats, arguments.clip_sigma))
            stacked = torch.cat(views, dim=0)
            mu_all, logvar_all = model.encode(stacked)
            rows = len(index)
            z_views = [mu_all[i * rows:(i + 1) * rows, :model.z_inv_dim] for i in range(len(views))]
            contrastive = (multi_view_contrastive_loss(z_views) if len(views) > 1
                           else mu_all.new_zeros(()))
            recon = model.decode(mu_all)
            loss, recon_value, kl_value, _ = vae_oh_loss(
                recon, stacked, mu_all, logvar_all, contrastive, beta=arguments.beta,
                lambda_contrast=arguments.lambda_contrast, channel_weights=weights)
            if len(views) > 1 and (arguments.zinv_var_weight > 0 or arguments.zinv_cov_weight > 0):
                variance, covariance = zinv_variance_covariance(z_views, target_std=0.5)
                loss = (loss + arguments.zinv_var_weight * variance
                        + arguments.zinv_cov_weight * covariance)
            optimiser.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step(); scheduler.step(); step += 1

            if step % arguments.eval_every == 0 or step == arguments.ssl_steps:
                train_z = encode_all(model, data["train"][selection], stats,
                                     arguments.clip_sigma)[:, :model.z_inv_dim].numpy()
                val_z = encode_all(model, data["validation"], stats,
                                   arguments.clip_sigma)[:, :model.z_inv_dim].numpy()
                if step >= arguments.selection_warmup:
                    score, fraction = guarded_davies_bouldin(
                        train_z, val_z, seed, 10, arguments.min_cluster_fraction)
                else:
                    score, fraction = float("-inf"), 0.0
                diag = {}
                try:
                    from sklearn.metrics import adjusted_rand_score
                    km = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(
                        l2_normalize(train_z))
                    assignment = km.predict(l2_normalize(val_z))
                    diag = {"ari_vs_label": float(adjusted_rand_score(labels["validation"],
                                                                     assignment)),
                            "ari_vs_flow": float(adjusted_rand_score(flows["validation"],
                                                                     assignment))}
                except Exception:
                    diag = {}
                history.append({**diag,
                                "step": step, "score": None if score == float("-inf") else score,
                                "min_cluster_fraction": fraction,
                                "recon": float(recon_value), "kl": float(kl_value),
                                "zinv_std": float(mu_all[:, :model.z_inv_dim].detach().std(0).mean())})
                if score > best["score"]:
                    best = {"score": score, "step": step, "fraction": fraction,
                            "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                if step % (arguments.eval_every * 5) == 0:
                    print(f"  seed {seed} step {step:5d}  recon {float(recon_value):.4f}  "
                          f"DB {history[-1]['score']}  minclu {fraction:.3f}  "
                          f"ARI|label {history[-1].get('ari_vs_label', float('nan')):.4f}  "
                          f"ARI|flow {history[-1].get('ari_vs_flow', float('nan')):.4f}  "
                          f"(best step {best['step']})", flush=True)

        if best["state"] is None:
            raise RuntimeError("no checkpoint passed the Davies-Bouldin degeneracy guard")
        final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best["state"])
        ssl_minutes = (time.time() - started) / 60
        heldout = reconstruction_mse(model, apply_norm_stats_oh(
            data["validation"], stats, arguments.clip_sigma), weights)
        print(f"  seed {seed}: stage 1 done, selected step {best['step']} "
              f"(DB {best['score']:.4f}, min cluster {best['fraction']:.3f}), "
              f"held-out recon {heldout:.5f}, {ssl_minutes:.1f} min", flush=True)

        checkpoints = [("db", best["step"], best["state"])]
        if arguments.also_final:
            checkpoints.append(("final", arguments.ssl_steps, final_state))
        for checkpoint_name, checkpoint_step, state in checkpoints:
          model.load_state_dict(state)
          latents = {split: encode_all(model, data[split], stats, arguments.clip_sigma)
                     for split in data}
          for kind in arguments.heads.split(","):
              head_started = time.time()
              head, chosen, probability = train_head(
                  latents, labels, kind, latents["train"].shape[1], seed, device,
                  arguments.head_epochs, arguments.head_lr, arguments.head_batch,
                  arguments.dropout)
              truth = labels["test"]
              per_flow = {flow: float(f1_score(truth[flows["test"] == i],
                                               probability[flows["test"] == i] >= .5,
                                               zero_division=0))
                          for i, flow in enumerate(wanted)}
              entry = {"seed": seed, "head": kind, "checkpoint": checkpoint_name,
                       "checkpoint_step": checkpoint_step, "group": arguments.group,
                       "ssl_lr": arguments.ssl_lr, "ssl_selected_step": best["step"],
                       "ssl_db": best["score"], "ssl_min_cluster_fraction": best["fraction"],
                       "heldout_recon": heldout, "head_best_epoch": chosen["epoch"],
                       "validation_f1": chosen["f1"], "validation_ap": chosen["ap"],
                       "test_f1": float(f1_score(truth, probability >= .5, zero_division=0)),
                       "test_ap": float(average_precision_score(truth, probability)),
                       "test_f1_per_flow": per_flow,
                       "head_parameters": int(sum(p.numel() for p in head.parameters())),
                       "encoder_parameters": int(sum(p.numel() for p in model.encoder.parameters())),
                       "ssl_minutes": ssl_minutes,
                       "head_minutes": (time.time() - head_started) / 60}
              results.append(entry)
              print(f"    [{checkpoint_name}@{checkpoint_step}] head={kind:6s} val F1 {chosen['f1']:.4f} @ {chosen['epoch']}  "
                    f"TEST F1 {entry['test_f1']:.4f} AP {entry['test_ap']:.4f}  "
                    + "  ".join(f"{k} {v:.4f}" for k, v in per_flow.items()) + "  "
                    f"head params {entry['head_parameters']:,}", flush=True)
              np.savez_compressed(output / f"test_seed{seed}_{kind}.npz",
                                  probability=probability, labels=truth, flow=flows["test"])
        (output / f"history_seed{seed}.json").write_text(json.dumps(history, indent=2),
                                                         encoding="utf-8")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (output / "summary.json").write_text(json.dumps(
        {"group": arguments.group, "ssl_lr": arguments.ssl_lr,
         "flow": arguments.flow or "both", "runs": results}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {output}", flush=True)


if __name__ == "__main__":
    main()
