"""What do the excluded lattice neighbours actually look like?

A Task4-c bundle is seeded on a 3x3x3 cubic lattice, and a line is lost either
at **seeding** (the seed left the candidate head region, or interpolated `oyf`
was not positive) or at **cleaning** (the integrated curve was non-finite, had
<= 2 points, degenerated on an axis, or its half arc length fell outside
[95%, 100.2%] of the requested `ds * maxiteration`).  Only survivors reach the
cache, so the excluded curves have to be regenerated from the raw VTK.

This script re-seeds the full 27-site lattice for chosen bundles, integrates
every site with the frozen RK45 settings, applies the cleaning rules, and draws
the centre line, the surviving face neighbours and the excluded ones together so
the difference is visible.  Exclusion cause is separated as:

* `oyf/domain`  -- fails the interpolated `oyf > 0` or inside-domain test;
* `arc length`  -- integrated but the half arc length missed the window;
* `short/nonfinite`, `degenerate` -- the other cleaning rejections;
* `head region` -- passes every test reproduced here, so the only remaining
  cause is the head-component membership test in `center_and_neighbors`.

It trains nothing and writes only figures plus a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments import Task4C_PhysicalLength_4_14 as builder  # noqa: E402
from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar  # noqa: E402
from FMT_Utils.Task4C_OctVAE_2_1 import (  # noqa: E402
    FACE_ORDER, PRIMITIVE_ORDER, lattice_codes, octahedral_selection,
)
from FMT_Utils.Task4C_PaperBundles_3_1 import load_flow  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LATTICE = np.array([[x, y, z] for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1)],
                   dtype=float)
CENTRE_SLOT = int(np.flatnonzero((LATTICE == 0).all(1))[0])
LATTICE[[0, CENTRE_SLOT]] = LATTICE[[CENTRE_SLOT, 0]]      # centre into slot 0, as the builder does
FACE_SET = {tuple(v) for v in FACE_ORDER}


def classify(flow_name, config, meta, row, grid, axes, oyf, present_sites):
    """Integrate all 27 lattice sites for one bundle and label each outcome."""
    centre = meta["center"][row]
    distance = float(meta["neighbor_distance"][row])
    seeds = centre + LATTICE * distance
    scale = {"ds": float(meta["ds"][row]), "maxiteration": int(meta["maxiteration"][row])}

    fluct, inside, _ = interpolate_scalar(seeds, axes, oyf)
    seed_ok = inside & (fluct > 0)

    lines, valid, arcs, steps, _ = builder.trace_lines(grid, seeds, scale, config)
    limit = scale["ds"] * scale["maxiteration"]
    low = config["integration"]["minimum_actual_half_arc_fraction"] * limit
    high = config["integration"]["maximum_actual_half_arc_fraction"] * limit

    records = []
    for slot, offset in enumerate(LATTICE):
        site = tuple(int(v) for v in offset)
        cause = None
        if site in present_sites:
            cause = "retained"
        elif not seed_ok[slot]:
            cause = "oyf/domain"
        elif np.any(arcs[slot] < low) or np.any(arcs[slot] > high):
            cause = "arc length"
        elif not valid[slot]:
            cause = "short/nonfinite or degenerate"
        else:
            cause = "head region"
        records.append({"slot": slot, "site": site, "is_face": site in FACE_SET,
                        "is_centre": site == (0, 0, 0), "cause": cause,
                        "curve": lines[slot] if valid[slot] else None,
                        "half_arcs": arcs[slot].tolist(),
                        "arc_fraction": float(arcs[slot].min() / limit) if limit else 0.0,
                        "steps": steps[slot].tolist()})
    return records, centre, distance


def draw(axis, records, centre, distance, title):
    styles = {"retained": ("#277da1", 1.6, "-"), "oyf/domain": ("#f94144", 1.3, "--"),
              "arc length": ("#f8961e", 1.3, "--"),
              "short/nonfinite or degenerate": ("#9b5de5", 1.3, "--"),
              "head region": ("#43aa8b", 1.3, "--")}
    for record in records:
        if not (record["is_face"] or record["is_centre"]):
            continue
        curve = record["curve"]
        if curve is None:
            seed = centre + np.asarray(record["site"], float) * distance
            axis.scatter(*((seed - centre) / distance), color="#f94144", marker="x", s=42)
            continue
        local = (curve - centre) / distance
        if record["is_centre"]:
            axis.plot(*local.T, color="#111111", linewidth=2.6, zorder=5)
            axis.scatter(*local[len(local)//2], color="#111111", s=26, zorder=6)
        else:
            colour, width, style = styles[record["cause"]]
            axis.plot(*local.T, color=colour, linewidth=width, linestyle=style, alpha=.95)
            seed = (centre + np.asarray(record["site"], float) * distance - centre) / distance
            axis.scatter(*seed, color=colour, s=18)
    axis.set_title(title, fontsize=7.5)
    axis.set_xlabel("x/h", fontsize=6); axis.set_ylabel("y/h", fontsize=6)
    axis.set_zlabel("z/h", fontsize=6)
    axis.tick_params(labelsize=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", default="channel", choices=["channel", "tbl"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--cache", default="outputs/exp_Task4C_LocalRebuild_1.0/physical")
    parser.add_argument("--config", default="config/exp_Task4C_LocalRebuild_1.0.json")
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--output", default="outputs/exp_Task4C_ExcludedLines_1.1")
    arguments = parser.parse_args()

    # load_spec merges flows/input_root/labels/bundles from the frozen base config
    config = builder.load_spec(str(ROOT / arguments.config))
    spec = [f for f in config["flows"] if f["name"] == arguments.flow][0]
    folder = ROOT / arguments.cache / arguments.flow / arguments.split
    seeds_cache = np.load(folder / "seeds.npy")
    with np.load(folder / "metadata.npz") as data:
        meta = {key: data[key] for key in data.files}
    codes, valid_rows = lattice_codes(seeds_cache, meta)
    select, usable = octahedral_selection(seeds_cache, meta)
    faces_missing = 6 - (select[:, 1:] >= 0).sum(axis=1)

    root = Path(config["input_root"])
    axes, _, _, oyf, grid, _ = load_flow(root / spec["flow"], spec["lambda2_threshold"])

    # pick a spread of examples: complete, and 1/2/3 missing faces
    rng = np.random.default_rng(7)
    chosen = []
    for want in (0, 1, 1, 2, 2, 3):
        pool = np.flatnonzero(faces_missing == want)
        if len(pool):
            chosen.extend(rng.choice(pool, min(2, len(pool)), replace=False).tolist())
    chosen = chosen[:arguments.examples]

    output = ROOT / arguments.output
    output.mkdir(parents=True, exist_ok=True)
    columns = 3
    rows_n = int(np.ceil(len(chosen) / columns))
    figure = plt.figure(figsize=(4.2 * columns, 3.8 * rows_n))
    summary = []
    for position, row in enumerate(chosen):
        present = {tuple(codes[row, j]) for j in np.flatnonzero(valid_rows[row])}
        records, centre, distance = classify(arguments.flow, config, meta, row, grid,
                                             axes, oyf, present)
        axis = figure.add_subplot(rows_n, columns, position + 1, projection="3d")
        causes = [r["cause"] for r in records if r["is_face"]]
        missing = sum(1 for c in causes if c != "retained")
        draw(axis, records, centre, distance,
             f"bundle {row}  label={int(meta['labels'][row])}  offset={distance/meta['local_grid_scale'][row]:.2f}h\n"
             f"{6-missing}/6 faces kept  " +
             ", ".join(sorted({c for c in causes if c != 'retained'})))
        summary.append({"row": int(row), "label": int(meta["labels"][row]),
                        "offset_over_h": float(distance / meta["local_grid_scale"][row]),
                        "faces_kept": int(6 - missing),
                        "face_causes": {str(r["site"]): r["cause"] for r in records if r["is_face"]},
                        "face_arc_fraction": {str(r["site"]): r["arc_fraction"]
                                              for r in records if r["is_face"]}})
    handles = [plt.Line2D([], [], color="#111111", lw=2.6, label="centre line"),
               plt.Line2D([], [], color="#277da1", lw=1.6, label="face neighbour kept"),
               plt.Line2D([], [], color="#f8961e", lw=1.3, ls="--", label="excluded: arc length"),
               plt.Line2D([], [], color="#43aa8b", lw=1.3, ls="--", label="excluded: head region"),
               plt.Line2D([], [], color="#9b5de5", lw=1.3, ls="--", label="excluded: short/degenerate"),
               plt.Line2D([], [], color="#f94144", lw=1.3, ls="--", label="excluded: oyf/domain (x = seed only)")]
    figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    figure.suptitle(f"Task4-c {arguments.flow}/{arguments.split}: excluded lattice neighbours "
                    f"against the centre line (axes in units of the lattice offset h)", fontsize=10)
    figure.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = output / f"excluded_{arguments.flow}_{arguments.split}.png"
    figure.savefig(path, dpi=150); plt.close(figure)
    (output / f"excluded_{arguments.flow}_{arguments.split}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    tally = {}
    for entry in summary:
        for cause in entry["face_causes"].values():
            tally[cause] = tally.get(cause, 0) + 1
    print("face-neighbour outcomes over the plotted examples:", tally)


if __name__ == "__main__":
    main()
