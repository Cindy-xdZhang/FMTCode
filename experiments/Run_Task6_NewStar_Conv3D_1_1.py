"""Conv3D voxel baseline on the re-seeded Task6 icosahedral stars.

Reuses the frozen Task4C spatial splat (`bundle_voxels`) and `Conv3DClassifier`
so the comparison against IcoVAE is against the repo's existing baseline family
rather than a new one. Stars are resampled 65 -> 32 points (the splat's fixed
length) and each bundle is centred and scaled into the unit cube, which is what
the splat asserts.
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
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from FMT_Utils.Task4C_HairpinBinary_2_1 import Conv3DClassifier  # noqa: E402
from FMT_Utils.Task4C_PaperBundles_3_1 import bundle_voxels  # noqa: E402
from FMT_Utils.IcosahedralGroup_3D import NEIGHBOUR_COUNT  # noqa: E402

RADII = (1.0, 2.0, 4.0, 8.0)


class WideConv3D(nn.Module):
    """Capacity-matched counterpart of the frozen ``Conv3DClassifier``.

    Same three-stage 3x3x3 / GroupNorm / MaxPool stack and the same
    ``AdaptiveAvgPool3d(2)`` read-out, with the channel widths and the hidden
    layer exposed so the baseline can be grown to the IcoVAE parameter count.
    At ``channels=(8,16,32), hidden=256`` it is the frozen model exactly.
    """

    def __init__(self, channels=(32, 64, 128), hidden=320, dropout=0.1, groups=4):
        super().__init__()
        c1, c2, c3 = channels
        self.convolution = nn.Sequential(
            nn.Conv3d(4, c1, 3, padding=1), nn.GroupNorm(groups, c1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(c1, c2, 3, padding=1), nn.GroupNorm(groups, c2), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(c2, c3, 3, padding=1), nn.GroupNorm(groups, c3), nn.ReLU(),
            nn.AdaptiveAvgPool3d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c3 * 8, hidden), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, voxels):
        return self.classifier(self.convolution(voxels)).squeeze(-1)


def load(cache, role, shells, scenes=None):
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
    curves, labels, scene = [], [], []
    for path in sorted(Path(cache).glob("*.npz")):
        with np.load(path, allow_pickle=False) as d:
            if str(d["role"]) != role:
                continue
            key = str(d["key"])
            if scenes and key.split(":")[0] not in scenes:
                continue
            if idx is None:
                idx = shell_index(d["radii"])
            curves.append(d["curves"][:, idx])
            labels.append(d["labels"])
            scene.extend([key.split(":")[0]] * len(d["labels"]))
    if not curves:
        raise FileNotFoundError(f"no {role} frames in {cache}")
    return np.concatenate(curves), np.concatenate(labels).astype(np.int64), np.array(scene)


def to_unit_bundles(curves):
    """[N,L,65,3] -> [N,L,32,3] centred on the bundle and scaled into [-1,1]."""
    x = torch.from_numpy(curves).float()
    n, lines, steps, _ = x.shape
    flat = x.permute(0, 1, 3, 2).reshape(n * lines, 3, steps)
    flat = F.interpolate(flat, size=32, mode="linear", align_corners=True)
    x = flat.reshape(n, lines, 3, 32).permute(0, 1, 3, 2).contiguous()
    centre = x.reshape(n, -1, 3).mean(1)[:, None, None, :]
    x = x - centre
    scale = x.reshape(n, -1).abs().amax(1).clamp_min(1e-12)[:, None, None, None]
    return (x / scale).clamp(-1.0, 1.0)


def macro_f1(pred, true):
    out = []
    for c in (0, 1):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/home/cheny1a/data/task6_icostar_1_1")
    p.add_argument("--output", default="outputs/exp_Task6_NewStar")
    p.add_argument("--label", default="conv3d")
    p.add_argument("--shells", default="1")
    p.add_argument("--scenes", default="")
    p.add_argument("--resolution", type=int, default=24)
    p.add_argument("--arch", default="frozen", choices=["frozen", "wide"])
    p.add_argument("--channels", default="32,64,128")
    p.add_argument("--hidden", type=int, default=320)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--schedule", default="cosine", choices=["constant", "cosine"])
    p.add_argument("--warmup-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=7068)
    args = p.parse_args()

    shells = tuple(float(x) for x in args.shells.split(","))
    scenes = tuple(x for x in args.scenes.split(",") if x)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_c, tr_y, _ = load(args.cache, "train", shells, scenes)
    te_c, te_y, te_s = load(args.cache, "test", shells, scenes)
    tr_x, te_x = to_unit_bundles(tr_c), to_unit_bundles(te_c)
    del tr_c, te_c
    lines = tr_x.shape[1]
    counts = torch.full((args.batch,), lines, device=device)

    if args.arch == "frozen":
        model = Conv3DClassifier().to(device)
    else:
        model = WideConv3D(tuple(int(x) for x in args.channels.split(",")),
                           hidden=args.hidden).to(device)
    n_par = int(sum(q.numel() for q in model.parameters()))
    print(f"  arch={args.arch} params={n_par:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_factor(step):
        if args.warmup_steps and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        if args.schedule == "cosine":
            t = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    tr_y_t = torch.from_numpy(tr_y).float().to(device)

    @torch.no_grad()
    def predict():
        model.eval()
        out = []
        for i in range(0, len(te_x), 512):
            b = te_x[i:i + 512].to(device)
            v = bundle_voxels(b, torch.full((len(b),), lines, device=device), args.resolution)
            out.append((model(v) > 0).long().cpu())
        model.train()
        return torch.cat(out).numpy()

    curve, best = [], {"f1": -1.0}
    t0 = time.time()
    g = torch.Generator(); g.manual_seed(args.seed)
    for step in range(args.steps):
        i = torch.randint(0, len(tr_x), (args.batch,), generator=g)
        b = tr_x[i].to(device)
        v = bundle_voxels(b, counts, args.resolution)
        loss = F.binary_cross_entropy_with_logits(model(v), tr_y_t[i.to(device)])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            pred = predict()
            f1 = macro_f1(pred, te_y)
            per = {s: macro_f1(pred[te_s == s], te_y[te_s == s]) for s in np.unique(te_s)}
            curve.append({"step": step + 1, "f1": f1, "ce": float(loss), "per_scene": per})
            if f1 > best["f1"]:
                best = {"f1": f1, "step": step + 1, "per_scene": per}
            print(f"  step {step+1:5d} f1 {f1:.4f} best {best['f1']:.4f}", flush=True)

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rec = dict(label=args.label, args=vars(args), shells=list(shells), lines=lines,
               train_n=int(len(tr_y)), test_n=int(len(te_y)),
               test_pos=float(te_y.mean()), final_f1=curve[-1]["f1"], best=best, curve=curve,
               params=n_par, arch=args.arch, seconds=time.time() - t0)
    (out / f"{args.label}.json").write_text(json.dumps(rec, indent=2))
    print(f"[done] {args.label} best {best['f1']:.4f} final {curve[-1]['f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
