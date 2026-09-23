"""Build 12-vertex icosahedral streamline stars on Task6_CorelineDataset_2.7_share.

Why this exists instead of ``preintegrated/icosa_bundles_1.1``
-------------------------------------------------------------
The shipped preintegrated bundles select centres with ``IVD(v) > 0.8 * max``.
Measured over all 101 frames that rule puts *zero* positives in three of the six
scenes (SquareCylinder, cylinder3d, halfcylinderRe320: 620000 centres, 0
positives) and 82.7% positives in halfcylinderRe640, so a pooled
all-scene classifier would be degenerate.  The dataset README explicitly frees
collaborators from that rule -- "IVD候选区 ... 这是旧采样协议，不限制合作者重新
选择种子" -- while fixing the *label* rule, which we keep exactly: a centre is
positive iff its distance to the continuous coreline segments is < h.

Seeding is therefore stratified so every scene carries positives:

    positive   centre within h of a coreline      (on-tube)
    hard       centre in the shell [h, hard_max*h] around a coreline
    far        centre drawn from the old IVD > mean candidate region

The exact distance to the coreline is stored per sample so accuracy can later be
resolved against distance rather than only against the induced class balance.

Geometry: the twelve icosahedron *vertices* (not the 20 face centres used by the
preintegrated bundles), in one fixed global orientation so that the I_h action
permutes neighbour channels exactly.  Neighbours are traced at radii
1h, 2h, 4h, 8h, giving 1 + 12*4 = 49 streamlines per sample; a run may slice any
single shell or any multi-shell combination out of the cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from FMT_Utils.IcosahedralGroup_3D import icosahedron_vertices  # noqa: E402

RADII = (1.0, 2.0, 4.0, 8.0)


def _load_api(root: Path):
    root = str(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from task6_fields import Dataset  # noqa: E402
    return Dataset


def densify(corelines, step):
    """Resample polylines so nearest-point distance approximates segment distance."""
    out = []
    for c in corelines:
        if len(c) < 2:
            out.append(np.asarray(c, float))
            continue
        seg = np.diff(c, axis=0)
        lengths = np.linalg.norm(seg, axis=1)
        for i, l in enumerate(lengths):
            k = max(int(np.ceil(l / step)), 1)
            out.append(c[i] + np.outer(np.linspace(0.0, 1.0, k, endpoint=False), seg[i]))
        out.append(np.asarray(c[-1:], float))
    return np.concatenate(out, axis=0)


def segment_distance(points, corelines, chunk=8192):
    """Exact distance to the continuous coreline segments.

    The label rule is distance to the *continuous* polyline, not to its sampled
    vertices. Nearest-vertex distance agrees with the official labels on
    99.8653% of centres and a KD-tree over a h/10 densification on 99.9982%;
    both disagree only for points sitting within ~1e-4 h of the threshold.
    This computes the true point-to-segment distance, which reproduces the
    official labels exactly on all 101 frames / 1010000 centres.
    """
    best = np.full(len(points), np.inf)
    for c in corelines:
        c = np.asarray(c, float)
        if len(c) < 2:
            d = np.linalg.norm(points - c[0], axis=1)
            best = np.minimum(best, d)
            continue
        a, ab = c[:-1], np.diff(c, axis=0)
        denom = np.einsum("ij,ij->i", ab, ab)
        denom[denom == 0] = 1e-30
        for k in range(0, len(points), chunk):
            q = points[k:k + chunk]
            t = np.clip(np.einsum("qij,ij->qi", q[:, None, :] - a[None], ab) / denom, 0.0, 1.0)
            proj = a[None] + t[..., None] * ab[None]
            best[k:k + chunk] = np.minimum(
                best[k:k + chunk], np.linalg.norm(q[:, None, :] - proj, axis=-1).min(1))
    return best


def ball(rng, n, low, high):
    """Uniform in a spherical shell of inner radius ``low`` and outer ``high``."""
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = (low ** 3 + (high ** 3 - low ** 3) * rng.random(n)) ** (1.0 / 3.0)
    return d * r[:, None]


def draw_seeds(frame, tree, dense, ivd, rng, count, pos_frac, hard_frac, hard_max, margin,
               pos_scale=1.0):
    """Stratified centres, all far enough from the boundary to carry every neighbour."""
    lo = frame.bounds[:, 0] + margin
    hi = frame.bounds[:, 1] - margin
    if np.any(hi <= lo):
        return np.zeros((0, 3)), np.zeros(0)
    inside = np.all((dense >= lo) & (dense <= hi), axis=1)
    anchors = dense[inside]

    want = {"pos": int(round(count * pos_frac)),
            "hard": int(round(count * hard_frac))}
    want["far"] = max(count - want["pos"] - want["hard"], 0)
    if len(anchors) == 0:
        want = {"pos": 0, "hard": 0, "far": count}

    cells = None
    if want["far"]:
        zz, yy, xx = np.nonzero(ivd > ivd.mean())
        ax = frame.axes_xyz
        cand = np.stack([ax[0][xx], ax[1][yy], ax[2][zz]], axis=1)
        keep = np.all((cand >= lo) & (cand <= hi), axis=1)
        cells = cand[keep]
        if len(cells) == 0:
            cells = None

    got, kinds = [], []
    for kind in ("pos", "hard", "far"):
        need = want[kind]
        tries = 0
        while need > 0 and tries < 40:
            tries += 1
            n = max(need * 4, 256)
            if kind == "far":
                if cells is None:
                    break
                s = cells[rng.integers(0, len(cells), n)]
            else:
                a = anchors[rng.integers(0, len(anchors), n)]
                inner, outer = ((0.0, frame.h * pos_scale * 0.999) if kind == "pos"
                               else (frame.h * pos_scale, frame.h * hard_max))
                s = a + ball(rng, n, inner, outer)
            s = s[np.all((s >= lo) & (s <= hi), axis=1)]
            if not len(s):
                continue
            d, _ = tree.query(s, workers=-1)
            limit = frame.h * pos_scale
            s = s[d < limit] if kind == "pos" else s[d >= limit]
            if not len(s):
                continue
            s = s[:need]
            got.append(s)
            kinds.append(np.full(len(s), "pos hard far".split().index(kind)))
            need -= len(s)
    if not got:
        return np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(got, axis=0), np.concatenate(kinds)


def build_frame(frame, key, rng, args, offsets, reuse=None):
    ivd = np.load(frame.root / "ivd.npy")
    cores = frame.corelines()
    if not cores:
        return None
    dense = densify(cores, frame.h / 10.0)
    tree = cKDTree(dense)
    margin = frame.h * max(args.radii_tuple) * 1.05

    kept_mask = None
    if reuse is not None:
        centres, kinds = reuse[0], reuse[1]
        # A larger outer shell needs a larger boundary margin than the source cache
        # used; without this the neighbour seeds would be clipped, silently
        # distorting the star and breaking the exact group action.
        lo, hi = frame.bounds[:, 0] + margin, frame.bounds[:, 1] - margin
        inside = np.all((centres >= lo) & (centres <= hi), axis=1)
        centres, kinds = centres[inside], kinds[inside]
        kept_mask = inside
    else:
        centres, kinds = draw_seeds(frame, tree, dense, ivd, rng, args.count,
                                    args.pos_frac, args.hard_frac, args.hard_max, margin,
                                    args.positive_scale)
    if len(centres) == 0:
        return None

    seeds = (centres[:, None, :] + offsets[None, :, :] * frame.h).reshape(-1, 3)
    seeds = np.clip(seeds, frame.bounds[:, 0], frame.bounds[:, 1])
    res = frame.integrate(seeds, total_length=args.total_length,
                          points=args.points, threads=args.threads)
    lines = len(offsets)
    curves = res["curves"].reshape(len(centres), lines, args.points, 3)
    valid = res["valid"].reshape(len(centres), lines).all(axis=1)

    if kept_mask is not None:
        sub = np.flatnonzero(kept_mask)[valid]
        kept_mask = np.zeros(len(kept_mask), bool); kept_mask[sub] = True
    curves, centres, kinds = curves[valid], centres[valid], kinds[valid]
    if not len(centres):
        return None
    # Sampling above may use the KD-tree approximation, but the stored label and
    # distance are the exact point-to-segment values the official rule specifies.
    dist = segment_distance(centres, cores)
    labels = (dist < frame.h * args.positive_scale).astype(np.int64)
    return dict(curves=curves.astype(np.float32), centres=centres.astype(np.float32),
                labels=labels, dist=(dist / frame.h).astype(np.float32),
                kind=kinds.astype(np.int8), key=key, h=float(frame.h),
                kept_mask=kept_mask, valid_fraction=float(valid.mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/cheny1a/data/flowData3D/Task6_CorelineDataset_2.7_share")
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=3000)
    p.add_argument("--points", type=int, default=65)
    p.add_argument("--total-length", type=float, default=0.5)
    p.add_argument("--pos-frac", type=float, default=0.25)
    p.add_argument("--hard-frac", type=float, default=0.35)
    p.add_argument("--hard-max", type=float, default=8.0)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--split-file", default="",
                   help="JSON with train/test frame-key lists; defaults to the package split. "
                        "Use for a temporal split (early frames train, late frames test).")
    p.add_argument("--positive-scale", type=float, default=1.0,
                   help="positive iff distance to the continuous coreline < scale*h. The official "
                        "rule is scale=1; smaller values make the positive class tighter and rarer. "
                        "The on-tube seeding radius follows the same scale so positives stay trainable.")
    p.add_argument("--radii", default="1,2,4,8",
                   help="neighbour shell radii in units of h")
    p.add_argument("--official-centres", default="",
                   help="path to preintegrated/icosa_bundles_1.1; take centres (and therefore "
                        "positive examples) from the official bundles instead of re-seeding")
    p.add_argument("--centres-from", default="",
                   help="reuse the centres (and labels) of an existing cache, so a rebuild "
                        "changes only the integration and stays a controlled comparison")
    args = p.parse_args()

    root = Path(args.root)
    Dataset = _load_api(root)
    args.radii_tuple = tuple(float(x) for x in args.radii.split(","))
    ds = Dataset(str(root))
    split = ds.split
    if args.split_file:
        split = json.loads(Path(args.split_file).read_text())
    radii = tuple(float(x) for x in args.radii.split(","))
    verts = icosahedron_vertices()
    offsets = np.concatenate([np.zeros((1, 3))] + [verts * r for r in radii], axis=0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = [(role, k) for role in ("train", "test") for k in split[role]]
    plan = [x for i, x in enumerate(plan) if i % args.shards == args.shard]

    for role, key in plan:
        dest = out / f"{key.replace(':', '_')}.npz"
        if dest.exists():
            continue
        t0 = time.time()
        frame = ds.frame(key)
        digest = hashlib.sha256(f"{args.seed}:{key}".encode()).digest()[:8]
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        reuse = None
        if args.official_centres:
            r = ds.records[key]
            src = Path(args.official_centres) / "frames" / r["flow"] / f"frame_{r['index']:03d}"
            if not (src / "centres.npy").exists() and not (src / "centers.npy").exists():
                print(f"[skip] {key} (no official centres)", flush=True)
                continue
            name = "centers.npy" if (src / "centers.npy").exists() else "centres.npy"
            c0 = np.load(src / name).astype(np.float64)
            y0 = np.load(src / "labels.npy").astype(np.int64)
            reuse = (c0, np.full(len(c0), 3, np.int8), y0, None)
        elif args.centres_from:
            src = Path(args.centres_from) / dest.name
            if not src.exists():
                print(f"[skip] {key} (no source centres)", flush=True)
                continue
            with np.load(src, allow_pickle=False) as d0:
                reuse = (d0["centres"].astype(np.float64), d0["kind"], d0["labels"], d0["dist"])
        rec = build_frame(frame, key, rng, args, offsets, reuse)
        if rec is not None and reuse is not None and reuse[2] is not None and args.official_centres:
            # the recomputed exact-segment labels must reproduce the official ones
            kept = rec.get("kept_mask")
            if kept is not None and not np.array_equal(rec["labels"], reuse[2][kept]):
                raise ValueError(f"{key}: recomputed labels disagree with the official labels")
        if rec is None:
            print(f"[skip] {key}", flush=True)
            continue
        np.savez(dest, curves=rec["curves"], centres=rec["centres"], labels=rec["labels"],
                 dist=rec["dist"], kind=rec["kind"], role=role, key=key, h=rec["h"],
                 radii=np.array(radii), offsets=offsets, total_length=args.total_length,
                 positive_scale=args.positive_scale)
        print(f"[ok] {key:28s} role={role:5s} n={len(rec['labels']):5d} "
              f"pos={rec['labels'].mean():.4f} valid={rec['valid_fraction']:.3f} "
              f"{time.time()-t0:5.1f}s", flush=True)

    meta = dict(radii=list(radii), positive_scale=args.positive_scale,
                split_file=args.split_file or "package default", lines=len(offsets), points=args.points,
                total_length=args.total_length, count=args.count,
                pos_frac=args.pos_frac, hard_frac=args.hard_frac, hard_max=args.hard_max,
                geometry="icosahedron_vertices",
                label_rule=f"centre distance to continuous coreline segments < {args.positive_scale}*h")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
