"""Quantify the defect in the continuous-SO(3) view's nearest-vertex channel map.

A star indexed by twelve fixed channels cannot represent a fractional rotation, so
``IcosahedralGroup_3D.random_so3`` assigns channels by nearest icosahedral vertex.
This measures how often that assignment is not a permutation at all, which is the
case in which the augmented "star" contains duplicated and missing neighbours and
is therefore not a star any rotated flow could produce.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from FMT_Utils.IcosahedralGroup_3D import icosahedral_group, icosahedron_vertices  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/analysis_so3_channel_map.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    d = torch.as_tensor(icosahedron_vertices(), dtype=torch.float64)
    n = args.samples
    # the exact construction random_so3 uses
    q, r = torch.linalg.qr(torch.randn(n, 3, 3, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]
    flip = torch.where(torch.linalg.det(q) < 0, -1.0, 1.0)
    rot = torch.cat((q[:, :, :2], q[:, :, 2:] * flip[:, None, None]), dim=-1)

    cos = torch.einsum("nji,kj->nki", rot, d) @ d.T
    gather = cos.argmax(-1)
    ident = torch.arange(12)
    is_ident = (gather == ident).all(1)
    is_perm = (gather.sort(1).values == ident).all(1)
    ang = torch.rad2deg(torch.arccos(cos.gather(-1, gather[..., None]).squeeze(-1).clamp(-1, 1)))
    turn = torch.rad2deg(torch.arccos(((torch.diagonal(rot, dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))

    report = {
        "samples": n,
        "group_order_I": int(icosahedral_group("i")[0].shape[0]),
        "group_order_Ih": int(icosahedral_group("ih")[0].shape[0]),
        "identity_channel_map_fraction": float(is_ident.float().mean()),
        "bijective_fraction": float(is_perm.float().mean()),
        "non_bijective_fraction": float((~is_perm).float().mean()),
        "vertex_assignment_error_deg": {"mean": float(ang.mean()), "max": float(ang.max())},
        "rotation_when_channels_unchanged_deg": {
            "mean": float(turn[is_ident].mean()) if is_ident.any() else None,
            "max": float(turn[is_ident].max()) if is_ident.any() else None},
    }
    print(f"random SO(3) views, n={n}")
    print(f"  channel map is a permutation      : {report['bijective_fraction']*100:6.2f} %")
    print(f"  NOT a permutation (channels collide): {report['non_bijective_fraction']*100:6.2f} %")
    print(f"  identity map while vectors rotate : {report['identity_channel_map_fraction']*100:6.2f} %"
          f" (up to {report['rotation_when_channels_unchanged_deg']['max']:.1f} deg)")
    print(f"  vertex assignment error           : mean {ang.mean():.2f} deg, max {ang.max():.2f} deg")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
