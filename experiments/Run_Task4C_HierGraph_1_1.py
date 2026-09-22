"""Train the hierarchical FMT graph on the non-relaxed 193k Task4-c cache.

Same data, splits, decision threshold and evaluation as the p35 baselines, so
test F1 is directly comparable to the published .888404 (nearest6) / .893015
(fps6).  The FMT encoder stays training-free; only the graph and head are learnt.

Features are precomputed once into a shared cache and reused by every variant.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.Task4C_FeatureVariants_1_1 import variant_block_width
from FMT_Utils.Task4C_LearnedDescriptor_1_1 import MultiRootLearnedGraph
from FMT_Utils.Task4C_NeighborSelection_1_1 import neighbor_indices
from FMT_Utils.Task4C_OctVAE_2_1 import lattice_codes
from FMT_Utils.Task4C_HierGraph_1_1 import (  # noqa: E402
    HierarchicalFMTGraph, MultiRootFMTGraph, precompute, precompute_geometry,
    precompute_multiroot,
)

ROOT = Path(__file__).resolve().parents[1]
FLOWS = ("channel", "tbl")
KEYS = ("centre_inv", "centre_dir", "fps_nb", "fps_ctr", "fps_nn")
MULTIROOT_KEYS = ("centre_inv", "centre_dir", "nb", "mask", "orbit", "nbr_index")
LEARNED_KEYS = ("geometry", "centre_dir", "mask", "orbit", "nbr_index")


def load_multiroot(cache, split, device, feature_root, strategy,
                   derivative=True, transform="dft", frequencies=6,
                   descriptor="fmt"):
    """Per-root features for every valid line; cached in half precision."""
    parts, labels, flows = [], [], []
    tag = "" if (derivative and transform == "dft") else \
          f"_{'d' if derivative else 'nod'}{transform}"
    if frequencies != 6:
        tag += f"_k{frequencies}"
    source = Path(cache).parent.name
    if source not in ("exp_Task4C_BottomDensity_1.2_local", ""):
        tag += f"_{source}"
    for index, flow in enumerate(FLOWS):
        store = Path(feature_root) / f"mr_{strategy}{tag}_{flow}_{split}.pt"
        folder = Path(cache) / flow / split
        with np.load(folder / "metadata.npz") as data:
            meta = {key: data[key] for key in data.files}
        if descriptor == "transformer":
            store = store.with_name("geo_" + store.name)
        if store.exists():
            features = torch.load(store, map_location="cpu")
        elif descriptor == "transformer":
            features = precompute_geometry(
                np.load(folder / "geometry.npy"), meta["counts"], device)
            store.parent.mkdir(parents=True, exist_ok=True)
            torch.save(features, store)
        else:
            features = precompute_multiroot(
                np.load(folder / "geometry.npy"), np.load(folder / "seeds.npy"),
                meta["counts"], device, strategy=strategy,
                derivative=derivative, transform=transform, frequencies=frequencies)
            store.parent.mkdir(parents=True, exist_ok=True)
            torch.save(features, store)
        # Orbit class of each root's lattice site, recomputed rather than cached so
        # the existing half-precision feature stores stay valid.
        seed_array = np.load(folder / "seeds.npy")
        # Neighbour line indices for the fine level, recomputed (cheap: a cdist on
        # [B,27,3]) so the cached feature stores stay valid.
        features = dict(features)
        features["nbr_index"] = neighbor_indices(
            torch.as_tensor(seed_array, device=device, dtype=torch.float32),
            torch.as_tensor(np.asarray(meta["counts"]).astype(np.int64), device=device),
            strategy).to(torch.int16).cpu()
        codes, on_lattice = lattice_codes(seed_array, meta)
        orbit = np.where(on_lattice, np.abs(codes.astype(np.int64)).sum(-1), 4)
        features["orbit"] = torch.as_tensor(orbit, dtype=torch.int8)
        parts.append(features)
        labels.append(np.asarray(meta["labels"]).astype(np.int64))
        flows.append(np.full(len(labels[-1]), index))
    keys = LEARNED_KEYS if descriptor == "transformer" else MULTIROOT_KEYS
    merged = {k: torch.cat([p[k] for p in parts]) for k in keys}
    return merged, np.concatenate(labels), np.concatenate(flows), 0


def load(cache, split, device, feature_root):
    """Precomputed graph features for one split, cached on disk."""
    parts, labels, flows, fallback = [], [], [], 0
    for index, flow in enumerate(FLOWS):
        source = Path(cache).parent.name
        suffix = "" if source == "exp_Task4C_BottomDensity_1.2_local" else f"_{source}"
        store = Path(feature_root) / f"{flow}{suffix}_{split}.pt"
        folder = Path(cache) / flow / split
        with np.load(folder / "metadata.npz") as data:
            meta = {key: data[key] for key in data.files}
        if store.exists():
            blob = torch.load(store, map_location="cpu")
            features, missing = blob["features"], blob["fallback"]
        else:
            features, missing = precompute(
                np.load(folder / "geometry.npy"), np.load(folder / "seeds.npy"),
                meta["counts"], meta, device)
            store.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"features": features, "fallback": missing}, store)
        parts.append(features); fallback += missing
        labels.append(np.asarray(meta["labels"]).astype(np.int64))
        flows.append(np.full(len(labels[-1]), index))
    merged = {k: torch.cat([p[k] for p in parts]) for k in KEYS}
    return merged, np.concatenate(labels), np.concatenate(flows), fallback


def fit_stats(train):
    """Per-feature mean/std over the training split only, per block."""
    stats = {}
    for key, value in train.items():
        if not value.is_floating_point():
            continue
        if key == "geometry":
            # already normalised by `normalize_geometry(..., "max_radius")`; a
            # per-component rescale here would stretch the tracks anisotropically
            continue
        value = value.float()
        flat = value.reshape(-1, value.shape[-1])
        mean = flat.mean(0)
        std = flat.std(0).clamp_min(1e-6)
        stats[key] = (mean, std)
    return stats


def apply_stats(features, stats, device):
    out = {}
    for k, v in features.items():
        if k not in stats:
            out[k] = v.to(device)
        else:
            out[k] = ((v.float() - stats[k][0]) / stats[k][1]).to(device)
    return out


def evaluate(model, features, labels, batch, device):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(labels), batch):
            chunk = {k: v[start:start + batch] for k, v in features.items()}
            out.append(torch.softmax(model(chunk), -1)[:, 1].float().cpu())
    model.train()
    p = torch.cat(out).numpy()
    return (float(f1_score(labels, p >= .5, zero_division=0)),
            float(average_precision_score(labels, p)), p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="outputs/exp_Task4C_BottomDensity_1.2_local/physical")
    parser.add_argument("--features", default="outputs/exp_Task4C_HierGraph_features")
    parser.add_argument("--representation-dim", type=int, default=256)
    parser.add_argument("--fine-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--no-direction", action="store_true")
    parser.add_argument("--no-fine", action="store_true")
    parser.add_argument("--seeds", default="96611,96612,96613")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model", default="hier", choices=["hier", "multiroot"])
    parser.add_argument("--strategy", default="fps6",
                        choices=["fps6", "nearest6", "nearest3_fps3", "both"],
                        help="'both' concatenates the fps6 and nearest6 neighbour sets, "
                             "giving each root 12 neighbours to attend over; coverage was "
                             "the winning lever in round 3, so this pushes on it directly")
    parser.add_argument("--descriptor", default="fmt", choices=["fmt", "transformer"],
                        help="'transformer' learns the per-line block instead of DFT/DCT")
    parser.add_argument("--descriptor-dim", type=int, default=32)
    parser.add_argument("--descriptor-mode", default="pair", choices=["pair", "line"])
    parser.add_argument("--enc-chunk", type=int, default=None,
                        help="sequences per checkpointed encoder chunk (lower = less memory)")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="recompute descriptor activations in backward to fit large batches")
    parser.add_argument("--rot-aug", default="none", choices=["none", "oh", "so3"],
                        help="rotate each bundle during training (learned descriptor only)")
    parser.add_argument("--enc-layers", type=int, default=2)
    parser.add_argument("--enc-dim", type=int, default=64)
    parser.add_argument("--transform", default="dft", choices=["dft", "dct"])
    parser.add_argument("--frequencies", type=int, default=6,
                        help="frequencies in the invariant blocks (direction block stays at 6)")
    parser.add_argument("--no-derivative", action="store_true",
                        help="transform the raw track (root t=0 at origin) instead of dx/dt")
    parser.add_argument("--mr-fine", action="store_true",
                        help="give the multi-root model a second hop (fine level)")
    parser.add_argument("--schedule", default="constant",
                        choices=["constant", "cosine"])
    parser.add_argument("--root-dropout", type=float, default=0.0,
                        help="probability of hiding each valid root during training")
    parser.add_argument("--orbit", action="store_true",
                        help="embed each root's O_h orbit class (centre/face/edge/corner)")
    parser.add_argument("--pool", default="meanmax",
                        choices=["meanmax", "mean", "attn", "attnmax"])
    parser.add_argument("--label", default="base")
    parser.add_argument("--output", default="outputs/exp_Task4C_HierGraph_1.1")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, labels, flows, fallback = {}, {}, {}, {}
    for split in ("train", "validation", "test"):
        if arguments.model == "multiroot" and arguments.strategy == "both":
            a, l, fl, fb = load_multiroot(ROOT / arguments.cache, split, device,
                                          ROOT / arguments.features, "fps6",
                                          not arguments.no_derivative, arguments.transform,
                                          arguments.frequencies, arguments.descriptor)
            b, _, _, _ = load_multiroot(ROOT / arguments.cache, split, device,
                                        ROOT / arguments.features, "nearest6",
                                        not arguments.no_derivative, arguments.transform,
                                        arguments.frequencies, arguments.descriptor)
            f = dict(a); f["nb"] = torch.cat((a["nb"], b["nb"]), dim=2)
            f["nbr_index"] = torch.cat((a["nbr_index"], b["nbr_index"]), dim=2)
        elif arguments.model == "multiroot":
            f, l, fl, fb = load_multiroot(ROOT / arguments.cache, split, device,
                                          ROOT / arguments.features, arguments.strategy,
                                          not arguments.no_derivative, arguments.transform,
                                          arguments.frequencies, arguments.descriptor)
        else:
            f, l, fl, fb = load(ROOT / arguments.cache, split, device, ROOT / arguments.features)
        data[split], labels[split], flows[split], fallback[split] = f, l, fl, fb
    stats = fit_stats(data["train"])
    tensors = {s: apply_stats(data[s], stats, device) for s in data}
    print(f"[{arguments.label}] train {len(labels['train']):,}  val {len(labels['validation']):,}  "
          f"test {len(labels['test']):,}  centre-fallback bundles "
          f"{sum(fallback.values())}  repr {arguments.representation_dim}D "
          f"fine {arguments.fine_dim}D heads {arguments.heads} "
          f"direction={not arguments.no_direction} fine={not arguments.no_fine}", flush=True)

    output = ROOT / arguments.output
    output.mkdir(parents=True, exist_ok=True)
    results = []
    y = {s: torch.as_tensor(labels[s], device=device) for s in labels}

    for seed in [int(s) for s in arguments.seeds.split(",")]:
        torch.manual_seed(seed); np.random.seed(seed)
        if arguments.descriptor == "transformer":
            model = MultiRootLearnedGraph(
                seq_len=data["train"]["geometry"].shape[2],
                representation_dim=arguments.representation_dim,
                descriptor_dim=arguments.descriptor_dim, heads=arguments.heads,
                dropout=arguments.dropout, use_direction=not arguments.no_direction,
                pool=arguments.pool, use_orbit=arguments.orbit,
                root_dropout=arguments.root_dropout, d_model=arguments.enc_dim,
                enc_layers=arguments.enc_layers, mode=arguments.descriptor_mode,
                rot_aug=arguments.rot_aug,
                grad_checkpoint=arguments.grad_checkpoint,
                enc_chunk=arguments.enc_chunk).to(device)
            if arguments.rot_aug != "none" and "centre_dir" in stats:
                model.set_direction_stats(*stats["centre_dir"])
        elif arguments.model == "multiroot":
            model = MultiRootFMTGraph(
                representation_dim=arguments.representation_dim, heads=arguments.heads,
                dropout=arguments.dropout, use_direction=not arguments.no_direction,
                pool=arguments.pool, use_orbit=arguments.orbit,
                root_dropout=arguments.root_dropout, use_fine=arguments.mr_fine,
                fine_dim=arguments.fine_dim,
                block_width=variant_block_width(arguments.frequencies,
                                                arguments.transform)).to(device)
        else:
            model = HierarchicalFMTGraph(
                fine_dim=arguments.fine_dim, representation_dim=arguments.representation_dim,
                heads=arguments.heads, dropout=arguments.dropout,
                use_direction=not arguments.no_direction,
                use_fine=not arguments.no_fine).to(device)
        counts = model.parameter_counts()
        optimiser = torch.optim.AdamW(model.parameters(), lr=arguments.lr,
                                      weight_decay=arguments.weight_decay)
        # Best epochs land at 400-600 of 600 under a constant learning rate, i.e. the
        # run is still wandering when it stops.  Cosine decay to zero lets the last
        # third of the schedule settle instead of bouncing.
        schedule = None
        if arguments.schedule == "cosine":
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=arguments.epochs)
        generator = torch.Generator(device=device).manual_seed(seed)
        best = {"f1": -1.0, "ap": -1.0, "epoch": -1, "state": None}
        started = time.time()
        for epoch in range(arguments.epochs):
            order = torch.randperm(len(y["train"]), device=device, generator=generator)
            for start in range(0, len(order), arguments.batch):
                i = order[start:start + arguments.batch]
                chunk = {k: v[i] for k, v in tensors["train"].items()}
                loss = F.cross_entropy(model(chunk), y["train"][i])
                optimiser.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimiser.step()
            if schedule is not None:
                schedule.step()
            f1, ap, _ = evaluate(model, tensors["validation"], labels["validation"], 4096, device)
            if (f1, ap) > (best["f1"], best["ap"]):
                best = {"f1": f1, "ap": ap, "epoch": epoch,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            if epoch % 20 == 0 or epoch == arguments.epochs - 1:
                print(f"    seed {seed} epoch {epoch:3d}  loss {loss.item():.4f}  "
                      f"val F1 {f1:.4f} (best {best['f1']:.4f} @ {best['epoch']})", flush=True)
        model.load_state_dict(best["state"])
        f1, ap, probability = evaluate(model, tensors["test"], labels["test"], 4096, device)
        per_flow = {flow: float(f1_score(labels["test"][flows["test"] == i],
                                         probability[flows["test"] == i] >= .5,
                                         zero_division=0))
                    for i, flow in enumerate(FLOWS)}
        entry = {"label": arguments.label, "seed": seed, "model": arguments.model,
                 "strategy": arguments.strategy, "pool": arguments.pool,
                 "representation_dim": arguments.representation_dim,
                 "fine_dim": arguments.fine_dim, "heads": arguments.heads,
                 "use_direction": not arguments.no_direction,
                 "use_fine": not arguments.no_fine, "lr": arguments.lr,
                 "epochs": arguments.epochs, "best_epoch": best["epoch"],
                 "validation_f1": best["f1"], "validation_ap": best["ap"],
                 "test_f1": f1, "test_ap": ap, "test_f1_per_flow": per_flow,
                 "parameters": counts, "minutes": (time.time() - started) / 60}
        results.append(entry)
        print(f"    seed {seed}: val F1 {best['f1']:.4f} @ {best['epoch']}  "
              f"TEST F1 {f1:.4f}  AP {ap:.4f}  channel {per_flow['channel']:.4f} "
              f"tbl {per_flow['tbl']:.4f}  params {counts['total']:,}  "
              f"{entry['minutes']:.1f} min", flush=True)
        np.savez_compressed(output / f"test_{arguments.label}_seed{seed}.npz",
                            probability=probability, labels=labels["test"], flow=flows["test"])
        del model
        torch.cuda.empty_cache()

    scores = [r["test_f1"] for r in results]
    (output / f"summary_{arguments.label}.json").write_text(
        json.dumps({"label": arguments.label, "runs": results,
                    "test_f1_mean": float(np.mean(scores)),
                    "test_f1_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0},
                   indent=2), encoding="utf-8")
    print(f"\n[{arguments.label}] TEST F1 {np.mean(scores):.4f} "
          f"+- {np.std(scores, ddof=1) if len(scores)>1 else 0:.4f}  "
          f"(baseline nearest6 .8884, fps6 .8930)", flush=True)


if __name__ == "__main__":
    main()
