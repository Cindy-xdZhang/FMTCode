"""Render current-recipe reference / Raw / FMT triptychs without model files.

Reference reconstruction follows NetCDF_window_3D and compute_ivd_reference_3d,
with an explicit seed-label and p95-threshold equivalence check. Only one frame
is needed for the reference surface; no predicted surface is interpolated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp/task123_plotdeps"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
import netCDF4
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes

SKILL = Path(os.environ.get("NATURE_FIGURE_SKILL_ROOT", Path.home() / ".codex/skills/nature-figure"))
sys.path.insert(0, str(SKILL / "scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment

NAMES = {"cylinder3d": "Half-cylinder Re160", "tangaroa": "Tangaroa"}
METHODS = {
    "task1": ("Raw-PCA + KMeans", "FMT + KMeans"),
    "task2": ("Raw + VAE", "FMT + same VAE"),
    "task3": ("Raw-PCA residual", "Raw + FMT residual"),
}
COLORS = {"reference": "#DB9652", "correct": "#B63B36", "false_positive": "#8C4E93",
          "false_negative": "#D39B37", "geometry": "#B9BDC2"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference(metadata, seeds, labels, target):
    path = Path(metadata["source_path"])
    if not path.exists():
        basename = str(metadata["source_path"]).replace("\\", "/").rsplit("/", 1)[-1]
        path = Path("/home/zhanx0o/DeepVortex/FLowDataFolder") / basename
    if not path.exists():
        raise FileNotFoundError(path)
    identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()
                              + seeds.tobytes() + labels.tobytes()
                              + str(path.stat().st_mtime_ns).encode()).hexdigest()
    if target.exists():
        with np.load(target) as d:
            if "identity" in d.files and str(d["identity"]) == identity:
                return d["vertices"], d["faces"], d["bounds"], json.loads(str(d["audit_json"]))
    with netCDF4.Dataset(path) as ds:
        aliases = {"x": {"x", "xdim"}, "y": {"y", "ydim"},
                   "z": {"z", "zdim"}, "t": {"t", "time", "tdim"}}
        dims = {a: next(n for n in ds.dimensions if n.lower() in aliases[a]) for a in "xyzt"}
        sizes = {a: len(ds.dimensions[dims[a]]) for a in "xyzt"}
        strides = {a: max(1, int(np.ceil(sizes[a] / 96))) for a in "xyz"}
        assert strides == metadata["spatial_strides"]
        slices = {a: slice(0, sizes[a], strides[a]) for a in "xyz"}
        slices["t"] = slice(metadata["source_start_index"], metadata["source_start_index"]+1)
        canonical = [dims[a] for a in "tzyx"]
        names = next(names for names in (("u", "v", "w"), ("velocity_x", "velocity_y", "velocity_z"),
                                        ("Component1", "Component2", "Component3"))
                     if all(n in ds.variables for n in names))
        arrays = []
        for name in names:
            var = ds.variables[name]
            assert len(var.dimensions) == 4
            index = tuple(slices["tzyx"[canonical.index(dim)]] for dim in var.dimensions)
            data = np.ma.asarray(var[index]).filled(0.0)
            arrays.append(np.transpose(np.asarray(data, np.float32),
                          [var.dimensions.index(dim) for dim in canonical])[0])
        frame = np.stack(arrays, axis=-1)
        coords = {}
        for a in "xyzt":
            candidates = [dims[a]] + [n for n in ds.variables if n.lower() in aliases[a]]
            variable = next((ds.variables[n] for n in candidates if n in ds.variables
                             and ds.variables[n].dimensions == (dims[a],)), None)
            full = np.arange(sizes[a]) if variable is None else np.asarray(variable[:])
            coords[a] = full[slices[a]]
        assert np.isclose(coords["t"][0], metadata["source_time"])
    assert list(frame.shape) == metadata["loaded_shape_TZYXC"][1:]
    bounds = np.asarray([[coords[a][0] for a in "xyz"], [coords[a][-1] for a in "xyz"]], np.float32)
    counts = np.asarray(frame.shape[:3][::-1])
    # Match the original float32 vector-field spacing calculation.
    spacing = (bounds[1] - bounds[0]) / (counts - 1).astype(np.float32)
    dx, dy, dz = map(float, spacing)
    u, v, w = (frame[..., i] for i in range(3))
    wx = np.gradient(w, dy, axis=1) - np.gradient(v, dz, axis=0)
    wy = np.gradient(u, dz, axis=0) - np.gradient(w, dx, axis=2)
    wz = np.gradient(v, dx, axis=2) - np.gradient(u, dy, axis=1)
    wx -= wx.mean(); wy -= wy.mean(); wz -= wz.mean()
    ivd = np.sqrt(wx*wx + wy*wy + wz*wz).astype(np.float32)
    threshold = float(metadata["ivd_threshold"])
    rebuilt_threshold = float(np.percentile(ivd[np.isfinite(ivd)], 95))
    if not np.isclose(rebuilt_threshold, threshold, rtol=1e-7, atol=1e-7*max(1,abs(threshold))):
        raise ValueError(f"p95 threshold mismatch: {rebuilt_threshold} vs {threshold}")
    xyz = [np.linspace(bounds[0,i], bounds[1,i], counts[i]) for i in range(3)]
    at_seeds = RegularGridInterpolator(tuple(xyz[::-1]), ivd, bounds_error=True)(seeds[:,[2,1,0]]).astype(np.float32)
    mismatch = int(np.count_nonzero((at_seeds >= threshold) != labels))
    if mismatch:
        raise ValueError(f"reconstructed reference differs at {mismatch} seeds")
    spacing_xyz = [float(np.median(np.diff(a))) for a in xyz]
    verts_zyx, faces, _, _ = marching_cubes(ivd, threshold, spacing=tuple(spacing_xyz[::-1]))
    vertices = verts_zyx[:,[2,1,0]] + bounds[0]
    audit = {"source_index": metadata["source_start_index"], "source_time": metadata["source_time"],
             "ivd_percentile": 95, "threshold": threshold, "recomputed_threshold": rebuilt_threshold,
             "reference_label_mismatch": mismatch, "surface_source": "original velocity field; no label or prediction interpolation",
             "reference_sample_count": len(labels), "source_file_size": path.stat().st_size}
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, vertices=vertices, faces=faces, bounds=bounds,
                        identity=identity, audit_json=json.dumps(audit))
    return vertices, faces, bounds, audit


def draw(task, dataset, artifact, out, medium):
    with np.load(artifact) as d:
        seeds, labels = d["seeds"], d["reference"].astype(bool)
        preds = [d["raw_prediction"].astype(bool), d["fmt_prediction"].astype(bool)]
        info = json.loads(str(d["metadata_json"]))
    sidecar = json.loads(artifact.with_suffix(".json").read_text())
    assert sha(artifact) == sidecar["prediction_sha256"]
    assert all(p.shape == labels.shape for p in preds)
    vertices, faces, bounds, surface_audit = load_reference(info["metadata"], seeds, labels,
                         out/"reference_meshes"/f"{dataset}_{task}.npz")
    paper = medium == "paper"
    font = 7 if paper else 13
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": font, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.linewidth": .6, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(7.2, 2.9) if paper else (13.33, 5.4), facecolor="white")
    rectangles = [[.045+i*.32, .23, .28, .59] for i in range(3)]
    axes = [fig.add_axes(rect, projection="3d") for rect in rectangles]
    titles = ["IVD-p95 reference", *METHODS[task]]
    marker_size = 4.5 if paper else 12
    for i, ax in enumerate(axes):
        ax.set_proj_type("ortho")
        ax.view_init(elev=22 if dataset == "cylinder3d" else 23, azim=-62)
        span = bounds[1] - bounds[0]
        ax.set_box_aspect(span, zoom=.92)
        for j, axis in enumerate((ax.xaxis, ax.yaxis, ax.zaxis)):
            axis.set_major_locator(MaxNLocator(nbins=2))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)
            axis.pane.fill = False
            axis.pane.set_edgecolor("#D0D0D0")
        ax.set(xlim=bounds[:,0], ylim=bounds[:,1], zlim=bounds[:,2])
        # A shared coordinate triad and numerical bounds replace overlapping
        # default 3D ticks, without changing the physical bounds or aspect.
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
        if i == 0:
            center = bounds.mean(axis=0)
            projected_center = np.asarray(proj3d.proj_transform(*center, ax.get_proj())[:2])
            origin = np.array([.12, .85])
            for j, name in enumerate("xyz"):
                endpoint = center.copy(); endpoint[j] += span[j]*.2
                projected = np.asarray(proj3d.proj_transform(*endpoint, ax.get_proj())[:2])
                direction = projected-projected_center
                direction /= np.linalg.norm(direction)
                tip = origin + direction*.10
                ax.annotate("",xy=tip,xytext=origin,xycoords="axes fraction",
                            arrowprops={"arrowstyle":"->","color":"#5F646A","lw":.6})
                label = origin + direction*.145
                ax.text2D(*label,name,transform=ax.transAxes,fontsize=font,
                          ha="center",va="center",color="#444444")
        if i == 0:
            mesh = Poly3DCollection(vertices[faces], facecolor=COLORS["reference"], edgecolor="none", alpha=.42)
            mesh.set_rasterized(True)
            ax.add_collection3d(mesh)
            pos = seeds[labels]
            ax.scatter(*pos.T, s=marker_size, c=COLORS["correct"], edgecolors="none", depthshade=False, rasterized=True)
        else:
            p = preds[i-1]
            groups = [(labels&p, "correct", "o"), (~labels&p, "false_positive", "^"),
                      (labels&~p, "false_negative", "x")]
            assert sum(int(mask.sum()) for mask,_,_ in groups) + int((~labels&~p).sum()) == len(labels)
            for mask, color, marker in groups:
                if mask.any():
                    points = seeds[mask]
                    ax.scatter(*points.T, s=marker_size*(1.2 if marker=="x" else 1), c=COLORS[color],
                               marker=marker, linewidths=.6 if marker=="x" else 0,
                               depthshade=False, rasterized=True)
        fig.text(rectangles[i][0], .86, "abc"[i], fontsize=font+2, fontweight="bold")
        fig.text(rectangles[i][0]+rectangles[i][2]/2, .86, titles[i], fontsize=font,
                 ha="center", va="baseline")
        if i:
            metric = info["seedwise_metrics"][("raw","fmt")[i-1]]["f1"]
            fig.text(rectangles[i][0]+rectangles[i][2]/2, .19, f"Slice F1 = {metric:.3f}", ha="center", fontsize=font)
    fig.text(.5, .96, f"Task {task[-1]}  |  {NAMES[dataset]}", ha="center", fontsize=font+1, fontweight="bold")
    handles = [Line2D([],[],marker=m,linestyle="none",color=COLORS[c],markersize=4 if paper else 7,label=l)
               for c,m,l in [("correct","o","Correct vortex"),("false_positive","^","False positive"),
                              ("false_negative","x","Missed vortex")]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.09), ncol=3,
               frameon=False, fontsize=font, handletextpad=.3, columnspacing=1.8)
    fig.text(.5,.06, f"t = {info['metadata']['source_time']:.3g}  |  n = {len(labels):,} pathline primitives", ha="center",fontsize=font)
    bound_text = "   ".join(f"{a} [{bounds[0,j]:.3g}, {bounds[1,j]:.3g}]" for j,a in enumerate("xyz"))
    fig.text(.5,.02,"Shared bounds: "+bound_text,ha="center",fontsize=font,color="#505050")
    stem = out/medium/f"{dataset}_{task}_triptych"
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(stem)+".alignment.json", tolerance_pt=1.5,
                                       gutter_tolerance_pt=1.5, strict=True,
                                       axes=axes, panel_ids=["a", "b", "c"], row_groups=[["a", "b", "c"]])
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"),dpi=400 if paper else 220)
    if paper:
        fig.savefig(stem.with_suffix(".tiff"), dpi=600,
                    pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    counts = {arm: {"true_positive": int((labels&p).sum()), "false_positive": int((~labels&p).sum()),
                    "false_negative": int((labels&~p).sum()), "omitted_true_negative": int((~labels&~p).sum())}
              for arm,p in zip(("raw","fmt"),preds)}
    audit = {"artifact_sha256": sha(artifact), "surface_audit": surface_audit, "confusion_counts":counts,
             "crop": "none; whole loaded physical domain", "camera": [22 if dataset=="cylinder3d" else 23,-62],
             "prediction_interpolation": False, "medium": medium, "protocol": info["protocol"],
             "minimum_font_pt":font, "source_count":len(labels), "excluded_from_metrics":0,
             "reference_display": "p95 surface and all reference-positive seeds",
             "uncertainty": "single preregistered seed and slice; not a seed-aggregate estimate"}
    stem.with_suffix(".json").write_text(json.dumps(audit, indent=2))
    with (out/f"{dataset}_{task}_source_data.csv").open("w",newline="") as handle:
        writer=csv.writer(handle);writer.writerow(["x","y","z","ivd_p95_reference","raw_prediction","fmt_prediction"])
        writer.writerows([*map(float,s),int(y),int(a),int(b)] for s,y,a,b in zip(seeds,labels,*preds))
    print(stem, flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-root",type=Path,default=ROOT/"outputs/Other_Task123_PaperTriptychs_1.1")
    parser.add_argument("--datasets",nargs="+",default=list(NAMES))
    parser.add_argument("--tasks",nargs="+",default=list(METHODS))
    parser.add_argument("--media",nargs="+",default=["paper","slides"])
    args=parser.parse_args()
    for ds in args.datasets:
        for task in args.tasks:
            for medium in args.media:
                draw(task,ds,args.output_root/f"{ds}_{task}_predictions.npz",args.output_root,medium)


if __name__ == "__main__":
    main()
