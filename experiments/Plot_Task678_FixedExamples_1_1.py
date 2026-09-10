"""Actual trajectories in fixed examples: identical view and limits across methods."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from experiments.Report_Task678_HanAndVector_1_1 import FLOW_NAMES


def export(fig,target):
    fig.canvas.draw()
    if os.getenv('NATURE_FIGURE_SCRIPTS'):
        sys.path.insert(0,os.environ['NATURE_FIGURE_SCRIPTS'])
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig,json_out=str(target)+'.alignment.json',overlay_svg=str(target)+'.alignment.svg',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    fig.savefig(str(target)+'.pdf');fig.savefig(str(target)+'.svg');fig.savefig(str(target)+'.png',dpi=600)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    manifest=json.loads((root/'fixed_examples.json').read_text());assert manifest['status']=='PASS'
    out=root/'figures';out.mkdir(exist_ok=True)
    contract=dict(question='What do corresponding trajectories look like for a fixed unseen primitive?',
        claim='Illustrative examples only; aggregate capability conclusions use all registered test data.',
        archetype='controlled comparison',backend='python',size_mm=[183,157],selection=manifest['selection'],
        normalization='Subtract the initial center and divide by the initial radius for every column and task.',
        exclusions='Only the first eight query particles and first primitive are displayed by fixed index; the quantitative study uses all particles and primitives.',
        display='Orthographic projection; every row includes all visible support and every displayed prediction in common unclipped cubic limits; no bounding box.',
        uncertainty='Single fixed seed illustration; all three seeds are included in the quantitative figures.')
    (out/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,
        'axes.titlesize':8,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
    columns=[('truth','Ground truth','#333333'),('fmt_all','Original FMT + network','#267B9E'),
             ('vector_fmt6','Vector FMT + network','#875A99'),('raw_positions','Raw + network','#777777')]
    all_views=[]
    for entry,name in zip(manifest['datasets'],FLOW_NAMES):
        path=root/entry['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        with np.load(path,allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
        origin,radius=data['origin'],float(data['radius'])
        fig=plt.figure(figsize=(183/25.4,157/25.4))
        axes=fig.subplots(3,4,subplot_kw={'projection':'3d'})
        fig.subplots_adjust(left=.065,right=.985,bottom=.14,top=.865,wspace=.025,hspace=.075)
        views=[]
        for row,task in enumerate(('Task6','Task7','Task8')):
            support=(data[task+'__support']-origin)/radius
            lines={key:(data[task+'__'+key]-origin)/radius for key,_,_ in columns}
            points=np.concatenate([support.reshape(-1,3)]+[v.reshape(-1,3) for v in lines.values()])
            assert np.isfinite(points).all()
            center=(points.min(0)+points.max(0))/2
            span=max(float(np.ptp(points,axis=0).max())*1.08,1e-6)
            lo,hi=center-span/2,center+span/2
            views.append(dict(task=task,limits_min=lo.tolist(),limits_max=hi.tolist(),elevation=22,azimuth=-65,projection='orthographic'))
            for col,(key,title,color) in enumerate(columns):
                ax=axes[row,col];ax.set_proj_type('ortho');ax.view_init(elev=22,azim=-65)
                for line in support:ax.plot(*line.T,color='#BFBFBF',lw=.35,alpha=.5)
                for line in lines[key]:
                    ax.plot(*line.T,color=color,lw=.8)
                    ax.scatter(*line[0],color=color,s=3,depthshade=False)
                ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2])
                ax.set_box_aspect((1,1,1));ax.set_axis_off()
                ax.text2D(.01,1.02,chr(97+4*row+col),transform=ax.transAxes,fontsize=9,fontweight='bold',ha='left',va='bottom')
                if row==0:ax.set_title(title,pad=14,fontsize=7)
            label={'Task6':'Task6: local query','Task7':'Task7: hidden region','Task8':'Task8: composition'}[task]
            ax=axes[row,0];pos=ax.get_position()
            fig.text(.017,pos.y0+pos.height/2,label,rotation=90,rotation_mode='anchor',ha='center',va='center',fontsize=8)
        fig.suptitle(name+': fixed test trajectories',x=.53,y=.98,fontsize=10,fontweight='bold')
        fig.text(.53,.925,'Ground truth and three frozen-token decoders',ha='center',fontsize=8)
        fig.text(.53,.087,'Same view and scale within each row · 8 fixed queries · thin gray: visible support',ha='center',fontsize=7)
        fig.text(.53,.050,'First test primitive; seed 9110 · Task6 shows stage 1 · Task8 uses predicted arrivals',ha='center',fontsize=7)
        export(fig,out/entry['dataset'])
        all_views.append(dict(dataset=entry['dataset'],source_sha256=entry['sha256'],views=views))
    contract['views']=all_views
    (out/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
    print('Exported nine fixed-example figures.',flush=True)


if __name__=='__main__':main()
