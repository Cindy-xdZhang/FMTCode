"""Why is each lattice neighbour excluded, and is the excluded curve any different?

The plots in `Visualize_Task4C_ExcludedLines_1_1.py` suggest the excluded lines
are ordinary, full-length curves.  This quantifies that over many bundles by
re-integrating the full 27-site lattice and attributing every missing face to a
cause, plus comparing geometric statistics of retained against excluded curves.

Bundles are grouped by `scale_id` so all seeds sharing one `(ds, maxiteration)`
go through a single `trace_lines` call.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import Task4C_PhysicalLength_4_14 as builder  # noqa: E402
from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar  # noqa: E402
from FMT_Utils.Task4C_OctVAE_2_1 import FACE_ORDER, lattice_codes  # noqa: E402
from FMT_Utils.Task4C_PaperBundles_3_1 import load_flow  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LATTICE = np.array([[x, y, z] for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1)], float)
_CENTRE = int(np.flatnonzero((LATTICE == 0).all(1))[0])
LATTICE[[0, _CENTRE]] = LATTICE[[_CENTRE, 0]]
FACE_SET = {tuple(v) for v in FACE_ORDER}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", default="channel", choices=["channel", "tbl"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--bundles", type=int, default=150)
    parser.add_argument("--cache", default="outputs/exp_Task4C_LocalRebuild_1.0/physical")
    parser.add_argument("--config", default="config/exp_Task4C_LocalRebuild_1.0.json")
    parser.add_argument("--output", default="outputs/exp_Task4C_ExcludedLines_1.1")
    arguments = parser.parse_args()

    config = builder.load_spec(str(ROOT / arguments.config))
    spec = [f for f in config["flows"] if f["name"] == arguments.flow][0]
    folder = ROOT / arguments.cache / arguments.flow / arguments.split
    seeds_cache = np.load(folder / "seeds.npy")
    with np.load(folder / "metadata.npz") as data:
        meta = {key: data[key] for key in data.files}
    codes, valid_rows = lattice_codes(seeds_cache, meta)

    root = Path(config["input_root"])
    axes, _, _, oyf, grid, _ = load_flow(root / spec["flow"], spec["lambda2_threshold"])

    rng = np.random.default_rng(11)
    chosen = rng.choice(len(meta["counts"]), min(arguments.bundles, len(meta["counts"])),
                        replace=False)
    by_scale = defaultdict(list)
    for row in chosen:
        by_scale[int(meta["scale_id"][row])].append(int(row))

    tally = defaultdict(int)
    stats = defaultdict(list)
    for scale_id, rows in sorted(by_scale.items()):
        ds = float(meta["ds"][rows[0]]); steps = int(meta["maxiteration"][rows[0]])
        scale = {"ds": ds, "maxiteration": steps}
        limit = ds * steps
        low = config["integration"]["minimum_actual_half_arc_fraction"] * limit
        high = config["integration"]["maximum_actual_half_arc_fraction"] * limit
        seeds = np.concatenate([meta["center"][r] + LATTICE * float(meta["neighbor_distance"][r])
                                for r in rows])
        lines, valid, arcs, _, _ = builder.trace_lines(grid, seeds, scale, config)
        fluct, inside, _ = interpolate_scalar(seeds, axes, oyf)
        seed_ok = inside & (fluct > 0)
        for j, row in enumerate(rows):
            block = slice(j * 27, (j + 1) * 27)
            present = {tuple(codes[row, k]) for k in np.flatnonzero(valid_rows[row])}
            centre_curve = None
            local_valid = valid[block]; local_lines = lines[block]
            for slot, offset in enumerate(LATTICE):
                site = tuple(int(v) for v in offset)
                if site == (0, 0, 0) and local_valid[slot]:
                    centre_curve = local_lines[slot]
            for slot, offset in enumerate(LATTICE):
                site = tuple(int(v) for v in offset)
                if site not in FACE_SET:
                    continue
                index = block.start + slot
                if site in present:
                    cause = "retained"
                elif not seed_ok[index]:
                    cause = "oyf_or_domain"
                elif np.any(arcs[index] < low) or np.any(arcs[index] > high):
                    cause = "arc_length"
                elif not valid[index]:
                    cause = "short_or_degenerate"
                else:
                    cause = "head_region"
                tally[cause] += 1
                if local_valid[slot] and centre_curve is not None:
                    curve = local_lines[slot]
                    group = cause
                    stats[f"{group}_arc_fraction"].append(float(arcs[index].min() / limit))
                    stats[f"{group}_length"].append(
                        float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()))
                    stats[f"{group}_dist_to_centre"].append(
                        float(np.linalg.norm(curve - centre_curve, axis=1).mean()))
                    span = np.ptp(curve, axis=0)
                    stats[f"{group}_bbox_diagonal"].append(float(np.linalg.norm(span)))

    total = sum(tally.values())
    print(f"{arguments.flow}/{arguments.split}: {len(chosen)} bundles, "
          f"{total} face-neighbour slots\n")
    print("| cause | slots | share of all faces | share of exclusions |")
    print("|---|---:|---:|---:|")
    excluded_total = total - tally["retained"]
    for cause in ("retained", "head_region", "oyf_or_domain", "arc_length",
                  "short_or_degenerate"):
        n = tally[cause]
        share = f"{n/excluded_total:.4f}" if cause != "retained" and excluded_total else "-"
        print(f"| {cause} | {n} | {n/total:.4f} | {share} |")
    groups = ("retained", "head_region", "oyf_or_domain", "arc_length")
    print("\n| statistic | " + " | ".join(groups) + " |")
    print("|---" * (len(groups) + 1) + "|")
    for key in ("arc_fraction", "length", "dist_to_centre", "bbox_diagonal"):
        cells = []
        for g in groups:
            values = stats.get(f"{g}_{key}", [])
            cells.append(f"{np.mean(values):.4f}" if values else "-")
        print(f"| {key} mean | " + " | ".join(cells) + " |")
    counts = [len(stats.get(f"{g}_dist_to_centre", [])) for g in groups]
    print("| (curves measured) | " + " | ".join(str(c) for c in counts) + " |")
    output = ROOT / arguments.output
    output.mkdir(parents=True, exist_ok=True)
    payload = {"flow": arguments.flow, "split": arguments.split,
               "bundles": int(len(chosen)), "face_slots": int(total),
               "causes": dict(tally),
               "statistics": {k: {"n": len(v), "mean": float(np.mean(v)),
                                  "std": float(np.std(v))} for k, v in stats.items()}}
    (output / f"causes_{arguments.flow}_{arguments.split}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {output}/causes_{arguments.flow}_{arguments.split}.json")


if __name__ == "__main__":
    main()
