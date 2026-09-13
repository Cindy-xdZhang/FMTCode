"""Paired F1 comparison for three distinct learning tasks; no row exclusion.

Question: how does objective neighbour encoding change fixed-task performance?
Evidence roles: training-free clustering, learned unsupervised representation,
and supervised transfer to unseen primitive scales. Quantitative grid; all ten
datasets and all five seeds. Points show means, bars one sample standard
deviation, faint points individual runs. No significance test is performed.
Python/matplotlib; 183 x 105 mm; editable PDF/SVG and 600-dpi PNG preview.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--skill-root',default=os.environ.get('NATURE_FIGURE_SKILL_ROOT'))
    args=parser.parse_args()
    if not args.skill_root:raise ValueError('Set the figure QA skill root.')
    sys.path.insert(0,str(Path(args.skill_root)/'scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    root=Path(args.root)
    audit=json.loads((root/'independent_audit.json').read_text())
    assert audit['status']=='PASS' and audit['metric_rows']==400
    summary=json.loads((root/'summary.json').read_text())
    control=summary['experiment'].endswith('1.2')
    rows=list(csv.DictReader((root/'per_run.csv').open()))
    datasets=['channel','cylinder3d','halfcylinderRe640','halfcylinderRe6400','tangaroa',
              'deltaWing_resampled','deltaWing_LBM','f22raptor','boeing747','smokeBuoyancy']
    names=['Channel','Half-cylinder Re160','Half-cylinder Re640','Half-cylinder Re6400',
           'Tangaroa','Delta wing (resampled)','Delta wing (LBM)','F-22','Boeing 747','Buoyant smoke']
    tasks=['Task1','Task2','Task5']
    titles=['Task1: direct clustering','Task2: unsupervised learning','Task5: unseen scales']
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,
                         'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
                         'axes.spines.right':False,'axes.linewidth':0.65,'legend.frameon':False})
    fig,axes=plt.subplots(1,3,figsize=(7.20472440945,4.13385826772),sharey=True,
                          gridspec_kw={'left':0.255,'right':0.985,'bottom':0.17,'top':0.80,'wspace':0.17})
    old_color,new_color='#737D85','#087E8B'
    source=[]
    for panel,(ax,task,title) in enumerate(zip(axes,tasks,titles)):
        for index,dataset in enumerate(datasets):
            values={a:np.asarray([float(r['f1']) for r in rows if r['task']==task and
                                 r['dataset']==dataset and r['arm']==a]) for a in ('old_fmt','fmt_all_v2')}
            assert all(len(v)==5 for v in values.values())
            means=[values[a].mean() for a in ('old_fmt','fmt_all_v2')]
            ax.plot(means,[index,index],color='#C8CDD1',lw=0.9,zorder=1)
            for arm,color,marker in [('old_fmt',old_color,'s'),('fmt_all_v2',new_color,'o')]:
                v=values[arm]
                ax.scatter(v,index+np.linspace(-0.09,0.09,5),s=5,color=color,alpha=0.28,edgecolors='none',zorder=2)
                ax.errorbar(v.mean(),index,xerr=v.std(ddof=1),fmt=marker,color=color,
                            markersize=3.3,markeredgewidth=0.5,capsize=1.5,elinewidth=0.8,zorder=3)
                source.append({'task':task,'dataset':dataset,'arm':arm,'n':5,
                               'f1_mean':float(v.mean()),'f1_sample_std':float(v.std(ddof=1))})
        ax.set_xlim(-0.04,1.04);ax.set_xticks([0,.25,.5,.75,1.0]);ax.set_xticklabels(['0','.25','.5','.75','1'])
        ax.set_ylim(9.6,-0.6);ax.set_xlabel('F1');ax.set_title(title,pad=9)
        ax.set_yticks(range(10));ax.set_yticklabels(names)
        ax.tick_params(axis='y',length=0,pad=5);ax.tick_params(axis='x',length=2.5)
        ax.spines['left'].set_visible(False)
        ax.set_axisbelow(True);ax.grid(axis='x',color='#E7EAED',linewidth=0.5)
        ax.text(-0.02,1.085,'abc'[panel],transform=ax.transAxes,fontsize=8,fontweight='bold',va='bottom')
    heading='Replacing the FMT core: paired comparison' if control else 'Objective neighbour encoding: paired comparison'
    fig.text(0.255,0.955,heading,fontsize=9,ha='left',va='top')
    handles=[Line2D([],[],color=old_color,marker='s',linestyle='none',markersize=4,label='Previous task recipe'),
             Line2D([],[],color=new_color,marker='o',linestyle='none',markersize=4,
                    label='v2 core + original blocks' if control else 'fmt_all_v2')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(0.255,0.925),ncol=2,handletextpad=0.5,columnspacing=2)
    fig.text(0.255,0.063,'Mean ± 1 s.d.; 5 paired seeds per dataset. Faint points: individual runs.',fontsize=6.5)
    fig.text(0.255,0.027,'Frozen benchmark replay. Task5 retains the same Raw coordinate branch.',fontsize=6.5)
    out=root/'figures';out.mkdir(exist_ok=True)
    base=out/'fmt_all_v2_comparison'
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(base)+'.alignment.json',
                                       overlay_svg=str(base)+'.alignment.svg',tolerance_pt=1.5,
                                       gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(f'{base}.pdf',facecolor='white')
    fig.savefig(f'{base}.svg',facecolor='white')
    fig.savefig(f'{base}.png',dpi=600,facecolor='white')
    plt.close(fig)
    with (out/'source_data.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(source[0]));writer.writeheader();writer.writerows(source)
    (out/'caption.md').write_text(
        'Paired comparison of previous task recipes and fmt_all_v2 on all ten 3D flow entries. '
        'Panels show clustering with a fixed FMT encoder (Task1), unsupervised variational autoencoder representations '
        '(Task2), and supervised evaluation at unseen primitive scales (Task5). Markers indicate means '
        'and horizontal bars one sample standard deviation over five paired random seeds; faint points '
        'show individual runs. F1 is the harmonic mean of precision and recall against frozen IVD p95 '
        'labels. IVD means instantaneous vorticity deviation. Splits and downstream hyperparameters '
        'are fixed, with cluster identities and supervised decisions calibrated on validation data. '
        'Task5 uses the same Raw backbone and matched residual-network width in both arms. '
        'These benchmarks were used previously; this is a diagnostic paired replay. All five runs '
        'for both plotted arms are retained; Raw controls are provided in the accompanying tables. '
        'No significance test is performed. Source data: source_data.csv and per_run.csv.\n',encoding='utf-8')
    if control:
        with (out/'caption.md').open('a',encoding='utf-8') as handle:
            handle.write('Core replacement retains kin4 for Tasks1/2 and gram2+kin6 for Task5. '
                         'The two Task5 arms both use338 input dimensions and136834 parameters. '
                         'Old/Raw Task1/2 results are reused after exact data/label identity checks; '
                         'Task2 seed pairs can run on different GPU models. This control was registered '
                         'after the standalone comparison had begun reporting results. The exact '
                         'objectivity certificate covers the v2 core, not the retained blocks or Raw branch.\n')


if __name__=='__main__':main()
