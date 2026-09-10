"""Plot all registered flows/seeds only after the final prediction replay passes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

NAMES = ['Half-cylinder Re160','Half-cylinder Re640','Half-cylinder Re6400','Tangaroa',
         'DeltaWing (resampled)','DeltaWing (LBM)','F22','Boeing 747','Smoke buoyancy']
STYLES = {'fmt_all':('#267B9E','o','FMT + network'), 'raw_positions':('#737373','s','Raw + network'),
          'coordinates_only':('#C57B48','^','No flow token'), 'affine':('#444444','d','Affine interpolation')}


def export(fig, target):
    fig.canvas.draw()
    if os.getenv('NATURE_FIGURE_SCRIPTS'):
        sys.path.insert(0, os.environ['NATURE_FIGURE_SCRIPTS'])
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig,json_out=str(target)+'.alignment.json',overlay_svg=str(target)+'.alignment.svg',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    fig.savefig(str(target)+'.pdf'); fig.savefig(str(target)+'.svg'); fig.savefig(str(target)+'.png',dpi=600)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',default='outputs/Verify_Task678_HanSampling_1.1')
    args=parser.parse_args(); root=Path(args.root)
    audit=json.loads((root/'independent_audit.json').read_text())
    spec=json.loads((root/'runtime_config.json').read_text())
    assert audit['status']=='PASS' and audit['metric_records']==2376
    metric_path=root/'metrics.csv'
    assert hashlib.sha256(metric_path.read_bytes()).hexdigest()==audit['metrics_sha256']
    rows=list(csv.DictReader(metric_path.open(encoding='utf-8')))
    out=root/'figures'; out.mkdir(parents=True,exist_ok=True)
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,
        'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.6,'legend.frameon':False,'savefig.facecolor':'white'})
    contract=dict(question='How accurately does the frozen FMT token support fitting and held-out flow-map queries?',
        archetype='quantitative grid',backend='python',exports=['PDF','SVG','PNG'],
        training='All nine flows; FMT and Raw; all three seeds and every stored training probe; supervised even times only',
        comparison='Every flow and all four arms on each registered test role; no removed failures',
        uncertainty='Training curves: mean and sample SD over three seeds; test charts: every seed as a tick and mean as a marker; affine has one deterministic result',
        scope='No claimed improvement over old cohorts; no significance test; actual inference must follow inspected results',
        transforms='Positive logarithmic error axes; no temporal curve smoothing; recorded exposure coordinates must match across seeds',
        source_metrics_sha256=audit['metrics_sha256'],runtime_commit=audit['git_commit'],size_mm=[183,157])
    (out/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
    # The fitting figure addresses optimization; zero-token and affine belong to held-out control figures.
    fig,axes=plt.subplots(3,3,figsize=(183/25.4,157/25.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.112,right=.98,bottom=.115,top=.84,wspace=.24,hspace=.40)
    values=[]; curve_rows=[]
    for index,(dataset,name,ax) in enumerate(zip(spec['datasets'],NAMES,axes.flat)):
        for task,color in [('Task6','#267B9E'),('Task7','#C57B48')]:
            for arm in ('fmt_all','raw_positions'):
                curves=[]; x=None
                for seed in spec['seeds']:
                    path=root/'evidence_metadata/runs'/dataset/arm/f'seed{seed}'/f'{task}_training.json'
                    data=json.loads(path.read_text())['learning_curve']
                    xx=np.array([d['equivalent_exposure'] for d in data])
                    yy=np.array([d['train_position_nrmse'] for d in data])
                    assert (np.diff(xx)>0).all() and np.isfinite(yy).all() and (yy>0).all()
                    if x is not None:
                        np.testing.assert_array_equal(x,xx)
                    x=xx; curves.append(yy)
                    curve_rows.extend(dict(dataset=dataset,task=task,arm=arm,seed=seed,exposure=float(a),error=float(b)) for a,b in zip(xx,yy))
                curves=np.asarray(curves); mean=curves.mean(0); sd=curves.std(0,ddof=1)
                # Positive log-space uncertainty: show individual curves when a symmetric SD reaches zero.
                ax.plot(x,mean,color=color,lw=1.2 if arm=='fmt_all' else .9,ls='-' if arm=='fmt_all' else '--',
                        label=f"{task}: {'FMT' if arm=='fmt_all' else 'Raw'}")
                if (mean-sd>0).all():
                    ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.12 if arm=='fmt_all' else .06,linewidth=0)
                    values.extend((mean-sd).tolist()); values.extend((mean+sd).tolist())
                else:
                    for y in curves:
                        ax.plot(x,y,color=color,lw=.4,alpha=.4,ls='-' if arm=='fmt_all' else '--')
                    values.extend(curves.flatten().tolist())
        ax.set_yscale('log'); ax.set_xlim(0,51); ax.set_xticks([0,10,20,30,40,50])
        ax.yaxis.set_minor_locator(NullLocator()); ax.grid(axis='y',color='#E6E6E6',lw=.45)
        ax.set_title(name,pad=8)
        ax.annotate(chr(97+index),xy=(0,1),xycoords='axes fraction',xytext=(-18,8),textcoords='offset points',
                    fontsize=9,fontweight='bold',ha='left',va='bottom')
    low=10**np.floor(np.log10(min(values))); high=10**np.ceil(np.log10(max(values)))
    ticks=10.**np.arange(int(np.log10(low)),int(np.log10(high))+1)
    for ax in axes.flat:
        ax.set_ylim(low,high); ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter([f'{v:g}' for v in ticks]))
    fig.suptitle('Dense-query training: fitting the supervised trajectories',x=.54,y=.985,fontsize=10,fontweight='bold')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.949),ncol=4,fontsize=7,handlelength=2.5,columnspacing=1.4)
    fig.text(.014,.475,'Training position error / initial radius',rotation=90,rotation_mode='anchor',ha='center',va='center',fontsize=8)
    fig.text(.54,.058,'Equivalent query exposure (passes)',ha='center',fontsize=8)
    fig.text(.54,.022,'Three seeds · fixed training probes · no validation or test values',ha='center',fontsize=7)
    export(fig,out/'training_fit')
    with (out/'training_curve_source.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(curve_rows[0])); w.writeheader(); w.writerows(curve_rows)
    titles={'query_test':'New particles in known primitives', 'primitive_test':'Flow-map queries in unseen primitives',
            'time_test':'Known-token queries in unseen time windows'}
    for role,title in titles.items():
        fig,axes=plt.subplots(1,3,figsize=(183/25.4,112/25.4),sharey=True)
        fig.subplots_adjust(left=.225,right=.98,bottom=.17,top=.80,wspace=.22)
        selected=[r for r in rows if r['role']==role and r['variant']=='normal']
        for j,(task,ax) in enumerate(zip(('Task6','Task7','Task8'),axes)):
            for k,(arm,(color,marker,label)) in enumerate(STYLES.items()):
                for i,dataset in enumerate(spec['datasets']):
                    vals=np.array([float(r['position_nrmse']) for r in selected if r['dataset']==dataset and r['arm']==arm and r['task']==task])
                    assert len(vals)==(1 if arm=='affine' else 3) and np.isfinite(vals).all() and (vals>0).all()
                    y=i+(k-1.5)*.18
                    ax.scatter(vals,np.full(len(vals),y),marker='|',s=20,color=color,alpha=.75,linewidths=.6)
                    ax.scatter([vals.mean()],[y],marker=marker,s=10,color=color,linewidths=0,label=label if i==0 else None,zorder=3)
            ax.set_xscale('log'); ax.set_ylim(8.65,-.65); ax.set_yticks(range(9),NAMES)
            ax.xaxis.set_minor_locator(NullLocator()); ax.grid(axis='x',color='#E6E6E6',lw=.45)
            ax.set_title({'Task6':'Task6: local query','Task7':'Task7: hidden region','Task8':'Task8: composition'}[task],pad=12)
            ax.annotate(chr(97+j),xy=(0,1),xycoords='axes fraction',xytext=(-14,12),textcoords='offset points',
                        fontsize=9,fontweight='bold',ha='left',va='bottom')
            ax.set_xlabel('Position error / radius',labelpad=7)
        fig.suptitle(title,x=.60,y=.987,fontsize=10,fontweight='bold')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,.94),ncol=2,fontsize=7,handlelength=1,columnspacing=2)
        fig.text(.60,.040,'Ticks: individual seeds; symbols: mean · lower is better',ha='center',fontsize=7)
        export(fig,out/role)
    print('Exported complete audited training and three held-out comparisons.',flush=True)


if __name__=='__main__':
    main()
