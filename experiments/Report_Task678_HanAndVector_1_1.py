"""Join independently audited common-cohort experiments; never select a winner."""
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

ARMS=['fmt_all','vector_fmt6','raw_positions','coordinates_only','affine','vector_affine']
STYLE={'fmt_all':('#267B9E','o','Original FMT + network'), 'vector_fmt6':('#875A99','s','Vector FMT + network'),
       'raw_positions':('#707070','^','Raw + network'), 'coordinates_only':('#C68B4F','v','No flow token'),
       'affine':('#333333','d','Raw interpolation'), 'vector_affine':('#5F9F9D','D','Fourier reconstruction')}
NORMALIZED=['position_nrmse','supervised_time_nrmse','unseen_time_nrmse','mean_position_error','centered_shape_error','pair_distance_error','endpoint_error']


def export(fig,target):
    fig.canvas.draw()
    if os.getenv('NATURE_FIGURE_SCRIPTS'):
        sys.path.insert(0,os.environ['NATURE_FIGURE_SCRIPTS'])
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig,json_out=str(target)+'.alignment.json',overlay_svg=str(target)+'.alignment.svg',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    fig.savefig(str(target)+'.pdf');fig.savefig(str(target)+'.svg');fig.savefig(str(target)+'.png',dpi=600)
    plt.close(fig)


def load(root, expected):
    report=json.loads((root/'independent_audit.json').read_text())
    cfg=json.loads((root/'runtime_config.json').read_text())
    assert report['status']=='PASS' and report['metric_records']==expected
    assert hashlib.sha256((root/'metrics.csv').read_bytes()).hexdigest()==report['metrics_sha256']
    assert hashlib.sha256((root/'runtime_config.json').read_bytes()).hexdigest()==report['config_sha256']
    rows=list(csv.DictReader((root/'metrics.csv').open(encoding='utf-8')))
    return rows,report,cfg


def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='outputs/Verify_Task678_HanSampling_1.1')
    p.add_argument('--vector',default='outputs/Verify_Task678_VectorFMT_1.1');p.add_argument('--output',default='outputs/Task678_HanAndVector_1.1')
    args=p.parse_args();base,vector,out=Path(args.base),Path(args.vector),Path(args.output)
    a,ar,ac=load(base,2376);b,br,bc=load(vector,999)
    assert bc['base_config_sha256']==ar['config_sha256'] and ac['decoder']==bc['decoder']
    assert ac['datasets']==bc['datasets'] and ac['seeds']==bc['seeds']
    rows=a+b;out.mkdir(parents=True,exist_ok=True)
    grouped={}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in ['dataset','arm','task','role','variant']),[]).append(row)
    means=[]
    for (dataset,arm,task,role,variant),values in grouped.items():
        expected=1 if arm in ('affine','vector_affine') else 3
        assert len(values)==expected
        entry=dict(dataset=dataset,family=ac['families'][dataset],arm=arm,task=task,role=role,variant=variant,
                   seeds=expected,parameters=int(values[0]['parameters']),token_bytes=int(values[0]['token_bytes']))
        for key in NORMALIZED+['physical_mean_position_error']:
            x=np.array([float(v[key]) for v in values]);assert np.isfinite(x).all() and (x>=0).all()
            entry[key]=float(x.mean());entry[key+'_seed_sd']=float(x.std(ddof=1)) if expected>1 else ''
        means.append(entry)
    save_csv(out/'per_flow.csv',means)
    aggregates=[]
    for arm,task,role,variant in sorted({tuple(r[k] for k in ['arm','task','role','variant']) for r in means}):
        values=[r for r in means if (r['arm'],r['task'],r['role'],r['variant'])==(arm,task,role,variant)]
        assert len(values)==9
        result=dict(arm=arm,task=task,role=role,variant=variant,flows=9)
        for metric in NORMALIZED:
            result['dataset_macro_'+metric]=float(np.mean([r[metric] for r in values]))
            result['family_macro_'+metric]=float(np.mean([np.mean([r[metric] for r in values if r['family']==family]) for family in set(ac['families'].values())]))
        seed_values={}
        for seed in ([0] if arm in ('affine','vector_affine') else ac['seeds']):
            items=[r for r in rows if (r['arm'],r['task'],r['role'],r['variant'],int(r['seed']))==(arm,task,role,variant,seed)]
            assert len(items)==9
            seed_values[str(seed)]=float(np.mean([float(r['position_nrmse']) for r in items]))
        result['position_nrmse_macro_seed_sd']=float(np.std(list(seed_values.values()),ddof=1)) if len(seed_values)>1 else ''
        aggregates.append(result)
    save_csv(out/'macro.csv',aggregates)
    paired=[]
    for dataset in ac['datasets']:
        for task in ('Task6','Task7','Task8'):
            for role in ('fit','query_test','primitive_test','time_test'):
                values={r['arm']:r for r in means if (r['dataset'],r['task'],r['role'],r['variant'])==(dataset,task,role,'normal')}
                for method in ('fmt_all','vector_fmt6','vector_affine'):
                    for control in ('raw_positions','coordinates_only','affine'):
                        row=dict(dataset=dataset,task=task,role=role,method=method,control=control)
                        for metric in NORMALIZED:
                            denominator=values[control][metric]
                            row[metric+'_difference']=values[method][metric]-denominator
                            row[metric+'_ratio']=values[method][metric]/denominator if denominator>0 else ''
                        paired.append(row)
    save_csv(out/'paired_differences.csv',paired)
    contract=dict(question='How do the original and directional Fourier tokens compare on held-out flow-map queries?',
        claim='Quantitative comparison only; capability conclusions must use the complete per-flow evidence',
        archetype='quantitative grid',panels='Task6 new material queries, Task7 hidden-region reconstruction, Task8 predicted-arrival composition',
        backend='python',size_mm=[183,100],data='All 9 flows, all registered seeds and all six arms; three test roles',
        mean='Equal-flow mean of position RMSE divided by the initial primitive radius',
        uncertainty='Every neural seed macro is shown as a small tick; connected marker is the three-seed mean. Deterministic interpolation has one result.',
        exclusions='Training and validation roles are kept in CSV and excluded only from this held-out comparison figure',
        transforms='Log error axis, positive guard; no curve fitting or smoothing',
        base_audit=ar,vector_audit=br)
    (out/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,
        'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'legend.frameon':False,'savefig.facecolor':'white'})
    width,height=183/25.4,100/25.4
    fig,axes=plt.subplots(1,3,figsize=(width,height),sharey=True)
    fig.subplots_adjust(left=.105,right=.98,bottom=.25,top=.70,wspace=.22)
    all_values=[]; plot_rows=[]
    for i,(task,ax) in enumerate(zip(('Task6','Task7','Task8'),axes)):
        for arm in ARMS:
            color,marker,label=STYLE[arm];values=[]
            for x,role in enumerate(('query_test','primitive_test','time_test')):
                seeds=[0] if arm in ('affine','vector_affine') else ac['seeds']
                y=[]
                for seed in seeds:
                    v=[float(r['position_nrmse']) for r in rows if (r['arm'],r['task'],r['role'],r['variant'],int(r['seed']))==(arm,task,role,'normal',seed)]
                    assert len(v)==9
                    y.append(float(np.mean(v)))
                    plot_rows.append(dict(task=task,arm=arm,role=role,seed=seed,macro_position_nrmse=y[-1]))
                assert np.isfinite(y).all() and (np.array(y)>0).all()
                all_values.extend(y);values.append(np.mean(y))
                ax.scatter(np.full(len(y),x),y,color=color,marker='_',s=25,linewidths=.7,zorder=3)
            ax.plot(range(3),values,color=color,marker=marker,ms=3,lw=1.3 if arm in ('fmt_all','vector_fmt6') else .9,
                    ls='--' if arm in ('affine','vector_affine') else '-',label=label)
        ax.set_yscale('log');ax.set_xlim(-.16,2.16)
        ax.set_xticks(range(3),['New\nparticles','New\nprimitives','New time\nwindows'])
        ax.yaxis.set_minor_locator(NullLocator());ax.grid(axis='y',color='#E5E5E5',lw=.45)
        ax.set_title({'Task6':'Task6: local query','Task7':'Task7: hidden region','Task8':'Task8: composition'}[task],pad=12)
        ax.annotate(chr(97+i),xy=(0,1),xycoords='axes fraction',xytext=(-18,12),textcoords='offset points',fontsize=9,fontweight='bold',ha='left',va='bottom')
    low=10**np.floor(np.log10(min(all_values)));high=10**np.ceil(np.log10(max(all_values)))
    ticks=10.**np.arange(int(np.log10(low)),int(np.log10(high))+1)
    for ax in axes:
        ax.set_ylim(low,high);ax.yaxis.set_major_locator(FixedLocator(ticks));ax.yaxis.set_major_formatter(FixedFormatter([f'{t:g}' for t in ticks]))
    axes[0].set_ylabel('Position error / radius',labelpad=8)
    fig.suptitle('Frozen Fourier tokens: held-out flow-map queries',x=.55,y=.987,fontsize=10,fontweight='bold')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,.925),ncol=3,fontsize=7,handlelength=2,columnspacing=1.4)
    fig.text(.55,.105,'Nine-flow means · small ticks: individual seeds · lower is better',ha='center',fontsize=7)
    fig.text(.55,.047,'Task8 passes predicted endpoints; all future segment tokens are known.',ha='center',fontsize=7)
    export(fig,out/'held_out_comparison')
    save_csv(out/'figure_source.csv',plot_rows)
    print('JOINED AUDIT PASS: 3375 evaluations; all paired tables and comparison figure exported.',flush=True)


if __name__=='__main__':main()
