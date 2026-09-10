"""Plot complete training curves without smoothing or substituting test errors."""
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    source=Path(args.snapshot)
    snapshot=json.loads(source.read_text(encoding='utf-8-sig'))
    order=['cylinder3d','halfcylinderRe640','halfcylinderRe6400','tangaroa','deltaWing_resampled','deltaWing_LBM','f22raptor','boeing747','smokeBuoyancy']
    names=['Half-cylinder Re160','Half-cylinder Re640','Half-cylinder Re6400','Tangaroa','DeltaWing (resampled)','DeltaWing (LBM)','F22','Boeing 747','Smoke buoyancy']
    selected={e['dataset']:e for e in snapshot['entries'] if e['condition']=='memorize16'}
    assert set(selected)==set(order) and all(e['complete'] for e in selected.values()), 'Wait for every registered small-set fit'
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    (root/'source_snapshot.json').write_bytes(source.read_bytes())
    contract=dict(question='Can the full frozen FMT input support fitting actual observed training trajectories?',
        archetype='quantitative grid',panel_role='Stratification across all nine registered flows; different flow conditions bound the fitting claim',
        backend='python',size_mm=[183,157],scope='Training fit only; one seed, 16 regions per flow; Task6 has two short segments, Task7 one hidden region',
        uncertainty='None: one optimization seed, curves are complete training-set errors, not estimates across independent replicates',
        data_selection='All nine memorize16 conditions, all recorded steps of both trained tasks; other conditions address the separate data-scale comparison',
        transformations='Logarithmic y axis; x in thousands of updates; no smoothing or resampling',
        exports=['PDF','SVG','PNG'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (root/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
    rows=[]
    for dataset in order:
        for task in ('Task6','Task7'):
            for point in selected[dataset]['tasks'][task]['curve']:
                assert point['population']=='all training particles and noninitial times'
                rows.append(dict(dataset=dataset,task=task,step=point['step'],train_position_nrmse=point['train_position_nrmse']))
    with (root/'training_curves.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.6,'legend.frameon':False,'savefig.facecolor':'white'})
    width_inches,height_inches=183/25.4,157/25.4
    fig,axes=plt.subplots(3,3,figsize=(width_inches,height_inches),sharex=True,sharey=True)
    fig.subplots_adjust(left=.104,right=.98,bottom=.115,top=.84,wspace=.24,hspace=.40)
    colors={'Task6':'#267B9E','Task7':'#CC713B'}
    yvalues=np.array([r['train_position_nrmse'] for r in rows]);assert np.isfinite(yvalues).all() and (yvalues>0).all()
    low=min(1e-4,10**np.floor(np.log10(yvalues.min())))
    high=max(100.,10**np.ceil(np.log10(yvalues.max())))
    for index,(dataset,name,ax) in enumerate(zip(order,names,axes.flat)):
        for task in ('Task6','Task7'):
            curve=selected[dataset]['tasks'][task]['curve']
            steps=np.array([p['step'] for p in curve]);error=np.array([p['train_position_nrmse'] for p in curve])
            assert (np.diff(steps)>0).all() and (error>0).all()
            ax.plot(steps/1000,error,color=colors[task],lw=1.2,ls='-' if task=='Task6' else '--',label=task)
            ax.plot(steps[-1]/1000,error[-1],marker='o' if task=='Task6' else 's',ms=2.5,color=colors[task])
        ax.axhline(.001,color='#707070',lw=.7,ls=':',zorder=0)
        ax.set_yscale('log');ax.set_ylim(low,high);ax.set_xlim(0,41)
        ax.set_xticks([0,10,20,30,40])
        ticks=[1e-4,1e-3,1e-2,1e-1,1,10,100]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(['0.0001','0.001','0.01','0.1','1','10','100']))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(axis='y',color='#E6E6E6',lw=.45,zorder=0)
        ax.set_title(name,pad=8)
        ax.annotate(chr(97+index),xy=(0,1),xycoords='axes fraction',xytext=(-18,8),textcoords='offset points',
                    fontsize=9,fontweight='bold',ha='left',va='bottom')
    fig.suptitle('Full FMT + neural network: small-set fitting',x=.53,y=.985,fontsize=11,fontweight='bold')
    handles,labels=axes[0,0].get_legend_handles_labels()
    handles.append(mpl.lines.Line2D([],[],color='#707070',lw=.8,ls=':'));labels.append('Training target: 0.001')
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.949),ncol=3,fontsize=8,handlelength=2.8,columnspacing=2)
    fig.text(.014,.475,'Training position error / initial radius',rotation=90,rotation_mode='anchor',ha='center',va='center',fontsize=8)
    fig.text(.54,.058,'Optimization updates (thousands)',ha='center',fontsize=8)
    fig.text(.54,.022,'16 regions per flow · one seed · all training particles and times · no test data',ha='center',fontsize=7)
    fig.canvas.draw()
    if os.getenv('NATURE_FIGURE_SCRIPTS'):
        sys.path.insert(0,os.environ['NATURE_FIGURE_SCRIPTS'])
    from audit_panel_alignment import require_matplotlib_panel_alignment
    target=root/'small_set_fitting'
    require_matplotlib_panel_alignment(fig,json_out=str(target)+'.alignment.json',
        overlay_svg=str(target)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    fig.savefig(f'{target}.pdf')
    fig.savefig(f'{target}.svg')
    fig.savefig(f'{target}.png',dpi=600)
    plt.close(fig)
    print('Exported all nine flows and',len(rows),'recorded training-curve points.')


if __name__=='__main__':
    main()
