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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import proj3d
import netCDF4
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes

SKILL = Path(os.environ.get("NATURE_FIGURE_SKILL_ROOT", Path.home() / ".codex/skills/nature-figure"))
sys.path.insert(0, str(SKILL / "scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment

NAMES = {"cylinder3d": "Half-cylinder Re160", "halfcylinderRe640": "Half-cylinder Re640",
         "halfcylinderRe6400": "Half-cylinder Re6400", "tangaroa": "Tangaroa",
         "boeing747": "Boeing 747", "deltaWing_LBM": "Delta-wing"}
VIEWS = {"cylinder3d":(22,-62), "halfcylinderRe640":(22,-62), "halfcylinderRe6400":(22,-62),
         "tangaroa":(23,-62), "boeing747":(21,-58), "deltaWing_LBM":(22,-58)}
SCENES = {}
METHODS = {
    "task1": ("Raw-PCA + KMeans", "FMT + KMeans"),
    "task2": ("Raw + VAE", "FMT + same VAE"),
    "task3": ("Raw-PCA residual", "Raw + FMT residual"),
}
COLORS = {"reference": "#DB9652", "correct": "#B63B36", "false_positive": "#8C4E93",
          "false_negative": "#D39B37", "geometry": "#B9BDC2"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_scene(out, dataset, task, metadata, seeds, labels):
    path = out/"display_assets"/f"{dataset}_{'task1' if task=='task1' else 'task23'}.npz"
    with np.load(path) as data:
        scene = {key:data[key] for key in data.files if key!='info_json'}
        info = json.loads(str(data['info_json']))
    assert np.array_equal(seeds, scene['seeds']), "Local reference and remote prediction seed mismatch"
    assert np.array_equal(labels, scene['reference']), "Reference label mismatch"
    for key in ['source_start_index', 'source_time', 'spatial_strides', 'loaded_shape_TZYXC', 'ivd_threshold']:
        assert metadata[key] == info['metadata'][key], f"Reference identity mismatch: {key}"
    assert info['surface_audit']['reference_label_mismatch'] == 0
    scene['info'] = info
    scene['sha256'] = sha(path)
    return scene


def draw_background(ax, scene, paper):
    ax.computed_zorder = False
    surface=Poly3DCollection(scene['vertices'][scene['faces']],facecolor=COLORS['reference'],
                            edgecolor='none',alpha=.20,zorder=1)
    surface.set_rasterized(True);ax.add_collection3d(surface)
    if len(scene['geometry']):
        triangles=scene['geometry']
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        light=np.array([.35,-.45,.82]);light/=np.linalg.norm(light)
        shade=.45+.35*np.abs(normals@light)
        colors=np.column_stack((shade*.75,shade*.82,shade*.90,np.full(len(shade),.96)))
        # Keep the verified solid readable under the dense explanatory paths.
        # This is an explicitly ordered context overlay, not depth inference.
        geo=Poly3DCollection(triangles,facecolor=colors,edgecolor='none',zorder=4)
        geo.set_rasterized(True);ax.add_collection3d(geo)
    norm=Normalize(0,scene['info']['paths']['duration'])
    for path,length in zip(scene['paths'],scene['lengths']):
        path=path[:length]
        if length<2:continue
        segments=np.stack((path[:-1,:3],path[1:,:3]),axis=1)
        line=Line3DCollection(segments,cmap='viridis',norm=norm,linewidths=.28 if paper else .52,
                              alpha=.50,zorder=3)
        line.set_array((path[:-1,3]+path[1:,3])*.5)
        line.set_rasterized(True);ax.add_collection3d(line)


def draw(task, dataset, artifact, out, medium):
    with np.load(artifact) as d:
        seeds, labels = d["seeds"], d["reference"].astype(bool)
        preds = [d["raw_prediction"].astype(bool), d["fmt_prediction"].astype(bool)]
        info = json.loads(str(d["metadata_json"]))
    sidecar = json.loads(artifact.with_suffix(".json").read_text())
    assert sha(artifact) == sidecar["prediction_sha256"]
    assert all(p.shape == labels.shape for p in preds)
    scene = load_scene(out,dataset,task,info['metadata'],seeds,labels)
    vertices,faces,bounds = scene['vertices'],scene['faces'],scene['bounds']
    surface_audit = scene['info']['surface_audit']
    mode_a = task == 'task1' 
    paper = medium == "paper"
    font = 7 if paper else 13
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": font, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.linewidth": .6, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(7.2, 2.9) if paper else (13.33, 5.4), facecolor="white")
    rectangles = [[.045+i*.32, .23, .28, .59] for i in range(3)]
    axes = [fig.add_axes(rect, projection="3d") for rect in rectangles]
    titles = (["Geometry + IVD + paths" if len(scene['geometry']) else "IVD + pathlines",
               "FMT + KMeans clusters", "FMT vs IVD-p95"] if mode_a else ["IVD-p95 reference", *METHODS[task]])
    marker_size = 4.5 if paper else 12
    for i, ax in enumerate(axes):
        ax.set_proj_type("ortho")
        ax.view_init(elev=VIEWS[dataset][0], azim=VIEWS[dataset][1])
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
        if mode_a and i == 0:
            draw_background(ax,scene,paper)
        elif mode_a and i == 1:
            for mask,color,size,alpha in [(~preds[1],"#2468B4",marker_size*.5,.22),
                                           (preds[1],COLORS['correct'],marker_size,.95)]:
                points=seeds[mask]
                ax.scatter(*points.T,s=size,c=color,alpha=alpha,linewidths=0,depthshade=False,rasterized=True)
        elif i == 0:
            mesh = Poly3DCollection(vertices[faces], facecolor=COLORS["reference"], edgecolor="none", alpha=.42)
            mesh.set_rasterized(True)
            ax.add_collection3d(mesh)
            pos = seeds[labels]
            ax.scatter(*pos.T, s=marker_size, c=COLORS["correct"], edgecolors="none", depthshade=False, rasterized=True)
        else:
            p = preds[1] if mode_a else preds[i-1]
            if mode_a:
                mesh=Poly3DCollection(vertices[faces],facecolor=COLORS['reference'],edgecolor='none',alpha=.14)
                mesh.set_rasterized(True);ax.add_collection3d(mesh)
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
        if mode_a and i == 1:
            fig.text(rectangles[i][0]+rectangles[i][2]/2,.19,"Red: vortex; blue: non-vortex",ha="center",fontsize=font)
        elif i:
            metric = info["seedwise_metrics"][("raw","fmt")[i-1]]["f1"]
            fig.text(rectangles[i][0]+rectangles[i][2]/2, .19, f"Slice F1 = {metric:.3f}", ha="center", fontsize=font)
    if mode_a:
        colorax=fig.add_axes([.105,.225,.16,.014])
        duration=scene['info']['paths']['duration']
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,duration),cmap='viridis'),
                        cax=colorax,orientation='horizontal',ticks=[])
        cb.outline.set_visible(False)
        fig.text(.185,.19,f"Path time: 0 to {duration:.3g}",ha='center',fontsize=font)
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
             "crop": "none; whole loaded physical domain", "camera": list(VIEWS[dataset]),
             "prediction_interpolation": False, "medium": medium, "protocol": info["protocol"],
             "minimum_font_pt":font, "source_count":len(labels), "excluded_from_metrics":0,
             "reference_display": "p95 surface; Mode a adds integrated paths and verified geometry" if mode_a else "p95 surface and all reference-positive seeds",
             "mode":"a" if mode_a else "b", "scene_sha256":scene['sha256'],
             "pathline_metadata":scene['info']['paths'] if mode_a else None,
             "simulation_geometry":scene['info']['geometry'] if mode_a else None,
             "context_layer_order":"IVD surface, paths, solid geometry; explicit explanatory overlays" if mode_a else None,
             "uncertainty": "single preregistered seed and slice; not a seed-aggregate estimate"}
    stem.with_suffix(".json").write_text(json.dumps(audit, indent=2))
    with (out/f"{dataset}_{task}_source_data.csv").open("w",newline="") as handle:
        writer=csv.writer(handle);writer.writerow(["x","y","z","ivd_p95_reference","raw_prediction","fmt_prediction"])
        writer.writerows([*map(float,s),int(y),int(a),int(b)] for s,y,a,b in zip(seeds,labels,*preds))
    print(stem, flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-root",type=Path,default=ROOT/"outputs/Other_Task123_PaperTriptychs_1.2")
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
