"""Audited error comparison and fixed-index geometry examples, Python backend only.

Figure question: does the repaired pipeline reconstruct the same held-out
trajectories below one initial-radius unit? One quantitative panel uses every
flow/seed; examples retain all seven lines and all 32 points for fixed IDs.
The legacy benchmark was already used, and the new encoder is explicitly named.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from FMT_Utils.PrimitiveVAE_3D import geometry_metrics

NAMES={"cylinder3d":"Cylinder Re160","halfcylinderRe640":"Cylinder Re640",
    "halfcylinderRe6400":"Cylinder Re6400","tangaroa":"Tangaroa",
    "deltaWing_resampled":"DeltaWing resampled","deltaWing_LBM":"DeltaWing LBM",
    "f22raptor":"F22","boeing747":"Boeing747","smokeBuoyancy":"Smoke buoyancy"}


def main(root,audit_scripts):
    sys.path.insert(0,str(audit_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    summary=json.loads((root/"summary.json").read_text())
    old=json.loads(Path("outputs/mainExp_Task6_PrimitiveVAE_2.1/summary.json").read_text())
    folder=root/"figures";folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":7,"axes.titlesize":7,"axes.labelsize":7,"xtick.labelsize":7,"ytick.labelsize":7,
        "legend.fontsize":7,"pdf.fonttype":42,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    def save(fig,name):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig,json_out=str(folder/f"{name}.alignment.json"),
            overlay_svg=str(folder/f"{name}.alignment.svg"),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
        fig.savefig(folder/f"{name}.pdf")
        fig.savefig(folder/f"{name}.svg")
        fig.savefig(folder/f"{name}.png",dpi=600)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,3.9))
    fig.subplots_adjust(left=.24,right=.97,bottom=.16,top=.84)
    prior={r["dataset"]:r for r in old["per_flow"]}
    for i,row in enumerate(summary["per_flow"]):
        p=prior[row["dataset"]]
        before=p["fmt_all_vae_test_mean"];after=row["signed_fmt_vae_legacy_test_mean"]
        if np.any(np.array([before,after])<=0):
            raise ValueError("Log-scale RMSE values must be positive")
        ax.plot([after,before],[i,i],color=".75",lw=.9,zorder=1)
        for value,spread,color,marker,label in (
            (before,p["fmt_all_vae_test_std"],"#8B8B8B","s","Original FMT + VAE (2.1)"),
            (after,row["signed_fmt_vae_legacy_test_std"],"#2475AC","o","Signed FMT + VAE (3.1)")):
            if value-spread<=0:
                raise ValueError("A standard-deviation interval crosses zero; use an individual-seed plot instead of truncating it")
            ax.errorbar(value,i,xerr=spread,fmt=marker,color=color,ms=4,lw=.8,capsize=2,label=label if i==0 else None,zorder=3)
    ax.axvline(1,color=".35",ls="--",lw=.8)
    ax.set_xscale("log");ax.set_yticks(range(9),[NAMES[r["dataset"]] for r in summary["per_flow"]]);ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value,position:f"{value:g}"))
    ax.set_xlabel("Position RMSE / initial neighbor radius (lower is better)")
    ax.legend(loc="lower left",bbox_to_anchor=(0,1.035),ncol=2,borderaxespad=0)
    fig.text(.24,.945,"Same historical test trajectories; mean ± standard deviation, 3 seeds",fontsize=7)
    save(fig,"legacy_test_reconstruction")
    source=np.load(root/"fixed_examples.npz")
    roles=[("truth","Ground truth","#242424"),("original_fmt","Original FMT + VAE","#8B8B8B"),
        ("signed_fmt_vae","Signed FMT + VAE","#2475AC"),("raw_vae","Raw + VAE","#69A6A1")]
    for dataset in NAMES:
        for sample,example_id in enumerate(source["example_ids"]):
            paths=[source[dataset+"__"+r][sample] for r,_,_ in roles]
            all_points=np.concatenate([x.reshape(-1,3) for x in paths])
            lower=all_points.min(0);upper=all_points.max(0)
            span=np.maximum(upper-lower,1e-2);lower-=span*.04;upper+=span*.04
            fig=plt.figure(figsize=(7.2,2.15));fig.subplots_adjust(left=.015,right=.985,bottom=.12,top=.83,wspace=.04)
            for col,((role,title,color),x) in enumerate(zip(roles,paths)):
                ax=fig.add_subplot(1,4,col+1,projection="3d")
                for line in range(7):
                    ax.plot(*x[line].T,color=color,lw=.9 if line==0 else .6,alpha=1 if line==0 else .75)
                    ax.scatter(*x[line,0],color=color,s=3)
                ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),zlim=(lower[2],upper[2]))
                ax.set_box_aspect(upper-lower);ax.view_init(elev=25,azim=-65);ax.set_axis_off()
                ax.text2D(.5,1.02,title,transform=ax.transAxes,ha="center",fontsize=7)
                score=geometry_metrics(x[None],paths[0][None])["position_rmse_r"]
                ax.text2D(.5,-.03,f"RMSE/r = {score:.3g}",transform=ax.transAxes,ha="center",fontsize=7)
            fig.text(.02,.95,f"{NAMES[dataset]} · fixed primitive {int(example_id)} · common view and coordinate limits",fontsize=8)
            save(fig,f"{dataset}_primitive_{int(example_id)}")
    notes=dict(backend="Python/matplotlib",archetype="quantitative comparison plus fixed-index geometry plates",
        metrics="same historical test, all nine flows and three seeds; mean and sample standard deviation",
        examples="indices 0,2000,4000 of legacy test, first preregistered old/new optimizer seed; no error-based selection",
        coordinates="unchanged initial-radius units; common bounds/view within each example; no alignment/permutation",
        statistics="no hypothesis tests; no independence claim for optimizer seeds or previously used temporal benchmark",
        exports="PDF and SVG with editable text, 600 dpi PNG; glyph floor 7 pt",
        method_scope="Paired historical comparison shows old/new FMT; Raw is in the geometry plates and full result table")
    (folder/"figure_contract.json").write_text(json.dumps(notes,indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=Path("outputs/mainExp_Task6_Reconstruction_3.1"))
    parser.add_argument("--audit-scripts",type=Path,required=True)
    args=parser.parse_args();main(args.root,args.audit_scripts)
