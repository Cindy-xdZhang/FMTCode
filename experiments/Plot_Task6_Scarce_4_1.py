"""Actual small-data reconstruction results: sample efficiency and paired controls.

Question: does the validation-selected FMT improve low-data test reconstruction?
Left: equal-flow error vs training count, retaining all three replicate seeds.
Right: all nine flows' paired changes against validation-selected Raw at n=1024.
Python/matplotlib only; actual audited metrics, no synthetic values or exclusions.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

NAMES={'cylinder3d':'Re160','halfcylinderRe640':'Re640','halfcylinderRe6400':'Re6400',
    'tangaroa':'Tangaroa','deltaWing_resampled':'DeltaWing resampled','deltaWing_LBM':'DeltaWing LBM',
    'f22raptor':'F22','boeing747':'Boeing747','smokeBuoyancy':'Smoke'}


def main(root,scripts):
    sys.path.insert(0,str(scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    spec=json.loads((root/'config.frozen.json').read_text())
    result=json.loads((root/'summary.json').read_text())
    assert result['audit']['passed']
    with (root/'metrics.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    index={(r['dataset'],int(r['train_size']),int(r['seed']),r['comparison']):float(r['position_rmse_r'])
        for r in rows if r['role']=='test' and int(r['scale_id'])==-1}
    controls={}
    for role in ['raw_frozen','raw_matched','raw_selected']:
        key=result['selected_candidates'][role]['id']
        controls.setdefault(key,[]).append(role)
    frequencies=result['selected_candidates']['fmt']['frequencies']
    lines=[('fmt',f'FMT ({frequencies} frequencies)','#2475AC','o')]
    palette=['#929292','#79A5A3','#C5996B']
    for (key,roles),color in zip(controls.items(),palette):
        names={'raw_frozen':'original','raw_matched':'matched','raw_selected':'selected'}
        label='Raw ('+' / '.join(names[r] for r in roles)+')'
        lines.append((roles[0],label,color,'s'))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
        'legend.fontsize':7,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig,(a,b)=plt.subplots(1,2,figsize=(7.2,3.9))
    fig.subplots_adjust(left=.10,right=.97,bottom=.17,top=.75,wspace=.85)
    for role,label,color,marker in lines:
        values=np.array([[np.mean([index[(d,n,s,role)] for d in spec['datasets']]) for s in spec['seeds']] for n in spec['train_sizes']])
        if not np.isfinite(values).all() or (values<=0).any():
            raise ValueError('Positive finite errors required for the log plot')
        a.plot(spec['train_sizes'],values.mean(1),label=label,color=color,marker=marker,ms=4,lw=1.0)
        for j,n in enumerate(spec['train_sizes']):
            a.scatter(np.full(3,n),values[j],color=color,s=8,alpha=.55,zorder=4)
    a.set_xscale('log');a.set_yscale('log')
    a.set_xticks(spec['train_sizes'],[str(n) for n in spec['train_sizes']])
    a.xaxis.set_minor_locator(NullLocator())
    a.set_yticks([.015,.02,.03,.04],['0.015','0.02','0.03','0.04'])
    a.yaxis.set_minor_locator(NullLocator())
    a.set_xlabel('Training primitives per flow')
    a.set_ylabel('Mean position RMSE / initial radius')
    a.set_title('a  Sample efficiency',loc='left',fontweight='bold',pad=8)
    paired=[]
    for i,d in enumerate(spec['datasets']):
        raw=np.array([index[(d,spec['primary_train_size'],s,'raw_selected')] for s in spec['seeds']])
        fmt=np.array([index[(d,spec['primary_train_size'],s,'fmt')] for s in spec['seeds']])
        assert (raw>0).all()
        change=100*(fmt/raw-1)
        paired.append(dict(dataset=d,seed_changes_percent=change.tolist(),mean=float(change.mean())))
        b.plot([change.min(),change.max()],[i,i],color='#A9BCCD',lw=.7)
        b.scatter(change,np.full(3,i),s=10,color='#2475AC',alpha=.55)
        b.scatter([change.mean()],[i],s=25,color='#2475AC',marker='D')
    b.axvline(0,color='.45',ls='--',lw=.7)
    b.set_yticks(range(9),[NAMES[d] for d in spec['datasets']]);b.invert_yaxis()
    b.set_xlabel('FMT error change vs selected Raw (%)')
    b.set_title('b  Paired comparison: 1024 samples',loc='left',fontweight='bold',pad=8)
    fig.legend(handles=[Line2D([],[],color=color,marker=marker,lw=1,ms=4,label=label) for _,label,color,marker in lines],
        loc='upper center',bbox_to_anchor=(.5,.985),ncol=2)
    fig.text(.10,.825,'Means and all 3 seeds; negative change favors FMT',fontsize=7)
    folder=root/'figures';folder.mkdir(exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(folder/'sample_efficiency.alignment.json'),
        overlay_svg=str(folder/'sample_efficiency.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(folder/'sample_efficiency.pdf',dpi=600)
    fig.savefig(folder/'sample_efficiency.svg',dpi=600)
    fig.savefig(folder/'sample_efficiency.png',dpi=600)
    plt.close(fig)
    (folder/'contract.json').write_text(json.dumps(dict(
        backend='Python/matplotlib',archetype='two complementary quantitative panels',
        conclusion='Test whether the fixed FMT candidate improves low-data reconstruction; do not presume the answer',
        statistics='All nine flows, all three sample sizes, all three subset/optimizer seeds. Means plus every seed; right range line is min-max, not a confidence interval.',
        comparison='Raw original recipe, FMT-matched regularization, and independently validation-selected Raw; identical Raw candidates merged into one curve',
        primary_size=spec['primary_train_size'],test='Previously used 3.1 benchmark; selection is validation-only',
        selected_candidates=result['selected_candidates'],paired=paired,
        review='Requires actual PDF text/collision checks and visual inspection before delivery'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/Verify_Task6_ScarceGeneralization_4.1'))
    p.add_argument('--audit-scripts',type=Path,required=True);args=p.parse_args();main(args.root,args.audit_scripts)
