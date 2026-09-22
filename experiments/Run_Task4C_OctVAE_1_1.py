"""Train the transferred octahedral SIREN-VAE encoder on Task4-c Hairpin bundles.

Keeps c156's evaluation contract: train-only statistics, checkpoint chosen on
validation F1 with average precision as tiebreak, Hairpin probability >= .5, and
F1 computed on the two flows' test bundles merged.

See `FMT_Utils/Task4C_OctVAE_1_1.py` for why the group action here is a pure
rotation with no channel permutation, and why the split latent makes rotation
augmentation safe even though the Hairpin label is orientation-dependent.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.SirenVAE_3D import multi_view_contrastive_loss, zinv_variance_covariance
from FMT_Utils.Task4C_OctVAE_1_1 import (  # noqa: E402
    OctVAEBundleClassifier, build_groups, rotate_groups, rotation_set,
)

ROOT = Path(__file__).resolve().parents[1]
FLOWS = ("channel", "tbl")


def load_split(cache, split, device):
    geometry, seeds, counts, labels, flow_id = [], [], [], [], []
    for index, flow in enumerate(FLOWS):
        folder = Path(cache) / flow / split
        geometry.append(np.load(folder / "geometry.npy"))
        seeds.append(np.load(folder / "seeds.npy"))
        with np.load(folder / "metadata.npz") as meta:
            counts.append(meta["counts"]); labels.append(meta["labels"])
        flow_id.append(np.full(len(labels[-1]), index))
    pack = lambda arrays, dtype: torch.as_tensor(
        np.concatenate(arrays), dtype=dtype, device=device)
    return {"geometry": pack(geometry, torch.float32), "seeds": pack(seeds, torch.float32),
            "counts": pack(counts, torch.long), "labels": pack(labels, torch.long),
            "flow": np.concatenate(flow_id)}


def group_stats(signal, mask):
    """Scalar RMS for the centre channel and one shared for the six neighbours.

    The Task1 normaliser, restricted to valid lines.  Scalar divisors keep the
    normalisation exactly rotation-equivariant; a per-component statistic would
    not commute with the group action.
    """
    valid = signal[mask]
    centre = valid[:, :1].pow(2).mean().sqrt().clamp_min(1e-8)
    neighbour = valid[:, 1:].pow(2).mean().sqrt().clamp_min(1e-8)
    return centre.detach(), neighbour.detach()


def apply_stats(signal, stats):
    centre, neighbour = stats
    return torch.cat((signal[:, :, :1] / centre, signal[:, :, 1:] / neighbour), dim=2)


def evaluate(model, data, stats, batch, device):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(data["labels"]), batch):
            stop = start + batch
            signal, mask = build_groups(data["geometry"][start:stop], data["seeds"][start:stop],
                                        data["counts"][start:stop], device=device)
            logits, _ = model(apply_stats(signal, stats), mask)
            probabilities.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu())
    model.train()
    probability = torch.cat(probabilities).numpy()
    truth = data["labels"].cpu().numpy()
    return {"f1": float(f1_score(truth, probability >= 0.5, zero_division=0)),
            "ap": float(average_precision_score(truth, probability))}, probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="outputs/exp_Task4C_LocalRebuild_1.0/physical")
    parser.add_argument("--head", default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--seeds", default="96721,96722")
    parser.add_argument("--group", default="oh", choices=["oh", "o", "reflect", "none"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-warmup", type=int, default=200)
    parser.add_argument("--lambda-contrast", type=float, default=1.0)
    parser.add_argument("--zinv-var-weight", type=float, default=1.0)
    parser.add_argument("--zinv-cov-weight", type=float, default=0.04)
    parser.add_argument("--output", default="outputs/exp_Task4C_OctVAE_1.1")
    parser.add_argument("--tag", default=None)
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = ROOT / arguments.cache
    data = {split: load_split(cache, split, device)
            for split in ("train", "validation", "test")}
    matrices = torch.as_tensor(rotation_set(arguments.group), dtype=torch.float32, device=device)
    print(f"train {len(data['train']['labels'])}  validation {len(data['validation']['labels'])}"
          f"  test {len(data['test']['labels'])}  positives(train) "
          f"{int(data['train']['labels'].sum())}  group={arguments.group} ({len(matrices)})",
          flush=True)

    output = ROOT / arguments.output
    if arguments.tag:
        output = output.parent / f"{output.name}_{arguments.tag}"
    output.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in [int(s) for s in arguments.seeds.split(",")]:
        torch.manual_seed(seed); np.random.seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        model = OctVAEBundleClassifier(steps=data["train"]["geometry"].shape[2],
                                       head=arguments.head).to(device)
        counts = model.parameter_counts()

        # train-only normalisation statistics
        probe, probe_mask = build_groups(
            data["train"]["geometry"][:2048], data["train"]["seeds"][:2048],
            data["train"]["counts"][:2048], device=device)
        stats = group_stats(probe, probe_mask)
        del probe, probe_mask

        optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.lr,
                                      weight_decay=arguments.weight_decay)
        total = arguments.epochs * math.ceil(len(data["train"]["labels"]) / arguments.batch)
        warmup = int(arguments.lr_warmup)

        def schedule(step):
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        best = {"f1": -1.0, "ap": -1.0, "epoch": -1, "state": None}
        started, step = time.time(), 0

        for epoch in range(arguments.epochs):
            order = torch.randperm(len(data["train"]["labels"]), device=device,
                                   generator=generator)
            for start in range(0, len(order), arguments.batch):
                index = order[start:start + arguments.batch]
                signal, mask = build_groups(data["train"]["geometry"][index],
                                            data["train"]["seeds"][index],
                                            data["train"]["counts"][index], device=device)
                normalised = apply_stats(signal, stats)
                logits, mu = model(normalised, mask)
                loss = F.cross_entropy(logits, data["train"]["labels"][index])

                if len(matrices) > 1 and arguments.lambda_contrast > 0:
                    element = torch.randint(1, len(matrices), (len(index),), device=device,
                                            generator=generator)
                    moved = apply_stats(rotate_groups(signal, element, matrices), stats)
                    mu_moved = model.encode_groups(moved)
                    flat_mask = mask.reshape(-1)
                    views = [mu.reshape(-1, mu.shape[-1])[flat_mask][:, :model.z_inv_dim],
                             mu_moved.reshape(-1, mu_moved.shape[-1])[flat_mask][:, :model.z_inv_dim]]
                    loss = loss + arguments.lambda_contrast * multi_view_contrastive_loss(views)
                    if arguments.zinv_var_weight > 0 or arguments.zinv_cov_weight > 0:
                        variance, covariance = zinv_variance_covariance(views, target_std=0.5)
                        loss = (loss + arguments.zinv_var_weight * variance
                                + arguments.zinv_cov_weight * covariance)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step(); scheduler.step(); step += 1

            metrics, _ = evaluate(model, data["validation"], stats, 256, device)
            if (metrics["f1"], metrics["ap"]) > (best["f1"], best["ap"]):
                best = {**metrics, "epoch": epoch,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            if epoch % 5 == 0 or epoch == arguments.epochs - 1:
                print(f"  seed {seed} epoch {epoch:3d}  loss {loss.item():.4f}  "
                      f"val F1 {metrics['f1']:.4f} AP {metrics['ap']:.4f}  "
                      f"(best {best['f1']:.4f} @ {best['epoch']})", flush=True)

        model.load_state_dict(best["state"])
        test, probability = evaluate(model, data["test"], stats, 256, device)
        elapsed = time.time() - started
        per_flow = {}
        truth = data["test"]["labels"].cpu().numpy()
        for index, flow in enumerate(FLOWS):
            pick = data["test"]["flow"] == index
            per_flow[flow] = float(f1_score(truth[pick], probability[pick] >= 0.5,
                                            zero_division=0))
        entry = {"seed": seed, "head": arguments.head, "group": arguments.group,
                 "lr": arguments.lr, "epochs": arguments.epochs,
                 "best_epoch": best["epoch"], "validation_f1": best["f1"],
                 "validation_ap": best["ap"], "test_f1": test["f1"], "test_ap": test["ap"],
                 "test_f1_per_flow": per_flow, "minutes": elapsed / 60.0,
                 "parameters": counts}
        results.append(entry)
        np.savez_compressed(output / f"test_predictions_seed{seed}_{arguments.head}.npz",
                            probability=probability, labels=truth, flow=data["test"]["flow"])
        print(f"  seed {seed}: val F1 {best['f1']:.4f} @ epoch {best['epoch']}  "
              f"TEST F1 {test['f1']:.4f}  AP {test['ap']:.4f}  "
              f"channel {per_flow['channel']:.4f} tbl {per_flow['tbl']:.4f}  "
              f"{elapsed/60:.1f} min  params {counts['total']:,}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scores = [r["test_f1"] for r in results]
    summary = {"head": arguments.head, "group": arguments.group, "lr": arguments.lr,
               "test_f1_mean": float(np.mean(scores)), "test_f1_std": float(np.std(scores, ddof=1))
               if len(scores) > 1 else 0.0, "runs": results}
    (output / f"summary_{arguments.head}.json").write_text(json.dumps(summary, indent=2),
                                                           encoding="utf-8")
    print(f"\nhead={arguments.head}  test F1 {summary['test_f1_mean']:.4f} "
          f"+- {summary['test_f1_std']:.4f}  over {len(scores)} seeds -> {output}", flush=True)


if __name__ == "__main__":
    main()
