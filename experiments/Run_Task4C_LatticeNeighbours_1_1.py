"""FMT p35 + MLP with lattice face-6 neighbours, on the original vs relaxed cache.

The handoff's .888404 row is FMT p35 with **nearest-6** neighbours; .893015 is the
same encoder with FPS-6.  Both pick neighbours by distance and ignore that the
bundle was seeded on a 3x3x3 cubic lattice.  This script adds a third rule,
`lattice6`: take the six **face-adjacent lattice sites** in the canonical
`x+, x-, y+, y-, z+, z-` order, which is exactly Task1's octahedral star and the
only rule under which the full `O_h` action, rotation *and* channel permutation,
is defined (`docs/exp_Task4C_OctVAE.md` §7.1).

Everything else is held fixed: the same p35 feature (141-D, `max_radius`
normalisation, 6 frequencies) and the same MLP, so the comparison isolates

1. **the neighbour rule** -- lattice6 against nearest6/fps6, and
2. **the head rule** -- the original cache against the relaxed one.

`lattice6` yields exactly one token per bundle, because only the lattice centre
has all six faces inside a 3x3x3 cube; `nearest6`/`fps6` yield one token per
valid line and pool them.  That difference is itself informative: if p35 also
collapses on the single octahedral token, the weakness found in §9 belongs to the
primitive rather than to the learned encoder.
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

from FMT_Utils.FMT_P35_NormFrequency_3_1 import normalize_geometry, primitive_features  # noqa: E402
from FMT_Utils.Task4C_FPSAugmentSearch_1_1 import Residual, mlp  # noqa: E402
from FMT_Utils.Task4C_NeighborSelection_1_1 import neighbor_indices  # noqa: E402
from FMT_Utils.Task4C_OctVAE_2_1 import octahedral_selection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FLOWS = ("channel", "tbl")
FREQUENCIES = 6
FEATURE_DIM = 24 * FREQUENCIES - 3          # 141, the p35 token width


@torch.no_grad()
def tokens_pooled(geometry, counts, seeds, strategy, device, chunk=256):
    """One 141-D token per valid line (centre + 6 chosen neighbours), mean+max pooled."""
    out = []
    for start in range(0, len(geometry), chunk):
        g = torch.as_tensor(geometry[start:start + chunk], device=device)
        c = torch.as_tensor(counts[start:start + chunk], device=device, dtype=torch.long)
        s = torch.as_tensor(seeds[start:start + chunk], device=device)
        x, mask = normalize_geometry(g, c, "max_radius")
        index = neighbor_indices(s, c, strategy)
        centre = torch.arange(x.shape[1], device=device)[None, :, None].expand(len(x), -1, -1)
        ids = torch.cat((centre, index), -1)
        features = x.new_zeros((len(x), x.shape[1], FEATURE_DIM))
        bi, li = mask.nonzero(as_tuple=True)
        for begin in range(0, len(bi), 2048):
            bs, ls = bi[begin:begin + 2048], li[begin:begin + 2048]
            features[bs, ls] = primitive_features(x[bs[:, None], ids[bs, ls]].squeeze(1),
                                                  FREQUENCIES)
        weight = mask[..., None].to(features.dtype)
        mean = (features * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        maximum = features.masked_fill(~mask[..., None], float("-inf")).amax(1)
        out.append(torch.cat((mean, torch.nan_to_num(maximum, neginf=0.0)), -1).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def tokens_lattice(geometry, counts, seeds, meta, device, chunk=256):
    """One 141-D token per bundle, from the octahedral star; unusable bundles dropped."""
    select, usable = octahedral_selection(seeds, meta)
    rows = np.flatnonzero(usable)
    out = []
    for start in range(0, len(rows), chunk):
        take = rows[start:start + chunk]
        g = torch.as_tensor(geometry[take], device=device)
        c = torch.as_tensor(counts[take], device=device, dtype=torch.long)
        x, _ = normalize_geometry(g, c, "max_radius")
        picks = torch.as_tensor(select[take], device=device, dtype=torch.long)
        star = torch.gather(x, 1, picks[:, :, None, None].expand(-1, -1, 32, 3))
        out.append(primitive_features(star, FREQUENCIES).float().cpu())
    return torch.cat(out), rows


def build(cache, split, strategy, device, match_lattice=False):
    """Encode one split.

    `match_lattice` restricts a pooled strategy (nearest6/fps6/nearest3_fps3) to
    exactly the bundles that `lattice6` can use -- those whose centre and all six
    face-adjacent lattice sites survived.  That removes the two confounds in a
    direct lattice6-vs-nearest6 comparison: lattice6 trains on ~33% of the
    bundles and is scored on the matching test subset.  With this flag both arms
    see identical bundles in both splits, so any remaining difference is the
    primitive itself.
    """
    features, labels, flows = [], [], []
    for index, flow in enumerate(FLOWS):
        folder = Path(cache) / flow / split
        geometry = np.load(folder / "geometry.npy")
        seeds = np.load(folder / "seeds.npy")
        with np.load(folder / "metadata.npz") as data:
            meta = {key: data[key] for key in data.files}
        counts = meta["counts"]
        if strategy == "lattice6":
            token, rows = tokens_lattice(geometry, counts, seeds, meta, device)
            label = np.asarray(meta["labels"])[rows]
        else:
            token = tokens_pooled(geometry, counts, seeds, strategy, device)
            label = np.asarray(meta["labels"])
            if match_lattice:
                _, usable = octahedral_selection(seeds, meta)
                keep = np.flatnonzero(usable)
                token = token[keep]; label = label[keep]
        features.append(token); labels.append(label.astype(np.int64))
        flows.append(np.full(len(label), index))
    return torch.cat(features), np.concatenate(labels), np.concatenate(flows)


def run(data, labels, seed, device, epochs, lr, batch, dropout):
    torch.manual_seed(seed); np.random.seed(seed)
    width = data["train"].shape[1]
    model = nn.Sequential(mlp(width, 256, dropout), Residual(256, dropout),
                          Residual(256, dropout), mlp(256, 128, dropout),
                          nn.Linear(128, 2)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mean = data["train"].mean(0, keepdim=True); std = data["train"].std(0, keepdim=True).clamp_min(1e-6)
    tensors = {k: ((v - mean) / std).to(device) for k, v in data.items()}
    y = torch.as_tensor(labels["train"], device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    best = {"f1": -1.0, "ap": -1.0, "epoch": -1, "state": None}
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(y), device=device, generator=generator)
        for start in range(0, len(order), batch):
            i = order[start:start + batch]
            loss = F.cross_entropy(model(tensors["train"][i]), y[i])
            optimiser.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimiser.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(tensors["validation"]), -1)[:, 1].cpu().numpy()
        metrics = (float(f1_score(labels["validation"], p >= .5, zero_division=0)),
                   float(average_precision_score(labels["validation"], p)))
        if metrics > (best["f1"], best["ap"]):
            best = {"f1": metrics[0], "ap": metrics[1], "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"]); model.eval()
    with torch.no_grad():
        p = torch.softmax(model(tensors["test"]), -1)[:, 1].cpu().numpy()
    return best, p, int(sum(q.numel() for q in model.parameters()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caches", default="original=outputs/exp_Task4C_LocalRebuild_1.0/physical,"
                                            "relaxed=outputs/exp_Task4C_RelaxedHead_1.0/physical")
    parser.add_argument("--strategies", default="nearest6,lattice6")
    parser.add_argument("--seeds", default="96611,96612")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--match-lattice-subset", action="store_true",
                        help="restrict pooled strategies to the bundles lattice6 can use, so "
                             "both arms see identical bundles in train and test")
    parser.add_argument("--output", default="outputs/exp_Task4C_LatticeNeighbours_1.1")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = ROOT / arguments.output; output.mkdir(parents=True, exist_ok=True)
    results = []
    for pair in arguments.caches.split(","):
        name, path = pair.split("=", 1)
        for strategy in arguments.strategies.split(","):
            started = time.time()
            data, labels, flows = {}, {}, {}
            for split in ("train", "validation", "test"):
                f, l, fl = build(ROOT / path, split, strategy, device,
                                 arguments.match_lattice_subset)
                data[split] = f; labels[split] = l; flows[split] = fl
            print(f"[{name}/{strategy}] token {data['train'].shape[1]}D  "
                  f"train {len(labels['train']):,}  test {len(labels['test']):,}  "
                  f"pos {labels['train'].mean():.4f}  ({time.time()-started:.0f}s to encode)",
                  flush=True)
            for seed in [int(s) for s in arguments.seeds.split(",")]:
                best, probability, parameters = run(
                    data, labels, seed, device, arguments.epochs, arguments.lr,
                    arguments.batch, arguments.dropout)
                truth = labels["test"]
                per_flow = {flow: float(f1_score(truth[flows["test"] == i],
                                                 probability[flows["test"] == i] >= .5,
                                                 zero_division=0))
                            for i, flow in enumerate(FLOWS)}
                entry = {"cache": name, "strategy": strategy, "seed": seed,
                         "match_lattice_subset": bool(arguments.match_lattice_subset),
                         "token_dim": int(data["train"].shape[1]),
                         "train_examples": int(len(labels["train"])),
                         "test_examples": int(len(truth)),
                         "validation_f1": best["f1"], "best_epoch": best["epoch"],
                         "test_f1": float(f1_score(truth, probability >= .5, zero_division=0)),
                         "test_ap": float(average_precision_score(truth, probability)),
                         "test_f1_per_flow": per_flow, "parameters": parameters}
                results.append(entry)
                print(f"    seed {seed}: val F1 {best['f1']:.4f}  TEST F1 {entry['test_f1']:.4f}  "
                      f"AP {entry['test_ap']:.4f}  channel {per_flow['channel']:.4f} "
                      f"tbl {per_flow['tbl']:.4f}  params {parameters:,}", flush=True)
    (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n| cache | strategy | train | test | mean test F1 |")
    print("|---|---|---:|---:|---:|")
    for name in {r["cache"] for r in results}:
        for strategy in arguments.strategies.split(","):
            subset = [r for r in results if r["cache"] == name and r["strategy"] == strategy]
            if subset:
                print(f"| {name} | {strategy} | {subset[0]['train_examples']:,} | "
                      f"{subset[0]['test_examples']:,} | "
                      f"{np.mean([r['test_f1'] for r in subset]):.4f} |")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
