"""Paired held-out gains and predetermined FTLE field examples, matplotlib only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.transforms import ScaledTranslation

FLOW_NAMES={'cylinder2d':'Cylinder','boussinesq':'Boussinesq',
            'pipedcylinder2d':'Piped cylinder','doublegyre2d':'Double gyre','macro':'Four-flow mean'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def panel_label(ax,label):
    ax.text(0,1,label,transform=ax.transAxes+ScaledTranslation(-15/72,6/72,ax.figure.dpi_scale_trans),
            fontsize=8,fontweight='bold',ha='left',va='bottom')


def export(fig,path,axes):
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=axes,json_out=str(path)+'.alignment.json',
                                      overlay_svg=str(path)+'.alignment.svg',tolerance_pt=1.5,
                                      gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(str(path)+'.pdf',dpi=600)
    fig.savefig(str(path)+'.svg',dpi=600)
    fig.savefig(str(path)+'.png',dpi=300)
    fig.savefig(str(path)+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)


def paired_values(summary,scale,baseline,flow):
    comparison=next(r for r in summary['test_comparisons'] if r['scale']==scale and r['baseline']==baseline)
    group=[r for r in comparison['paired'] if flow=='macro' or r['flow']==flow]
    seeds=sorted(set(r['seed'] for r in group))
    return seeds,[float(np.mean([r['psnr_gain'] for r in group if r['seed']==s])) for s in seeds]


def gains(root,out,summary,flows):
    arch=summary['selected']['architecture'];rows=[]
    fig,axs=plt.subplots(1,2,figsize=(180/25.4,83/25.4),sharey=True)
    fig.subplots_adjust(left=.16,right=.985,bottom=.25,top=.87,wspace=.30)
    colors={4:'#326A91',8:'#805E8E'}
    groups=[[(4,'espcn',-.09,'o'),(8,'espcn',.09,'s')],
            [(4,arch+'_none',-.21,'o'),(8,arch+'_none',-.07,'s'),
             (4,arch+'_raw',.07,'o'),(8,arch+'_raw',.21,'s')]]
    labels=['Beyond the scalar baseline','Contribution of the representation']
    for i,(ax,series) in enumerate(zip(axs,groups)):
        for scale,baseline,offset,marker in series:
            means,stds=[],[]
            for flow in flows+['macro']:
                seeds,values=paired_values(summary,scale,baseline,flow)
                assert len(seeds)==3
                means.append(np.mean(values));stds.append(np.std(values,ddof=1))
                rows.extend({'scale':scale,'baseline':baseline,'flow':flow,'seed':s,'psnr_gain_db':v} for s,v in zip(seeds,values))
            raw=baseline.endswith('_raw')
            label=f'{scale}×'+(' vs Raw' if raw else ' vs no geometry' if i else ' vs ESPCN')
            ax.errorbar(means,np.arange(len(flows)+1)+offset,xerr=stds,fmt=marker,
                        color=colors[scale],mfc='white' if raw else colors[scale],ms=3.8,
                        capsize=2,elinewidth=.7,label=label,zorder=3)
        ax.axvline(0,color='#888888',lw=.65,ls='--',zorder=1)
        ax.set_title(labels[i],fontsize=8,pad=11)
        ax.set_xlabel('Paired PSNR gain (dB)',labelpad=6)
        ax.set_yticks(np.arange(len(flows)+1),[FLOW_NAMES[f] for f in flows+['macro']])
        ax.set_ylim(len(flows)+.5,-.5);ax.grid(axis='y',color='#EEEEEE',lw=.6)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=6,
                  handlelength=1,columnspacing=1.1,labelspacing=.5)
        panel_label(ax,chr(97+i))
    fig.text(.16,.035,'Mean ± one standard deviation of three paired optimizer seeds; each run averages three test slices.',fontsize=6)
    export(fig,out/'paired_gains',list(axs))
    with (out/'paired_gains_source.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plate(root,out,summary,flow,scale,seed,source_root):
    arch=summary['selected']['architecture'];selected=summary['selected']['id']
    methods=[('espcn','ESPCN'),('unet','U-Net'),(arch+'_none','No geometry'),
             (arch+'_raw','Raw paths'),(selected,'P35 fusion')]
    folder=root/'runs'/f'{flow}_x{scale}_s{seed}'
    record=read(folder/selected/'result.json')
    first=sorted(r['file'] for r in record['metrics'] if r['split']=='test')[0]
    arrays={};hashes={}
    for method,_ in methods:
        path=folder/method/first
        with np.load(path) as ds:arrays[method]={k:ds[k] for k in ds.files}
        hashes[method]=sha(path)
    truth=arrays[selected]['truth'];mask=arrays[selected]['mask']
    for data in arrays.values():
        np.testing.assert_array_equal(data['truth'],truth);np.testing.assert_array_equal(data['mask'],mask)
    original=read(source_root/flow/'manifest.json');bounds=original['source']['bounds']
    row=next(r for r in original['records'] if r['file']==first)
    x0,x1,y0,y1=bounds;transpose=(y1-y0)>(x1-x0)
    extent=[y0,y1,x0,x1] if transpose else bounds
    xlabel,ylabel=('y','x') if transpose else ('x','y')
    width,height=extent[1]-extent[0],extent[3]-extent[2]
    transform=lambda a:a.T if transpose else a
    plot_width_mm=180*(.965-.11)/(2+.24)
    plot_height_mm=plot_width_mm*height/width
    row_gap_mm=8.
    figure_height=6*plot_height_mm+5*row_gap_mm+45
    fig,axes=plt.subplots(6,2,figsize=(180/25.4,figure_height/25.4))
    fig.subplots_adjust(left=.11,right=.965,bottom=30/figure_height,
                        top=1-15/figure_height,wspace=.24,hspace=row_gap_mm/plot_height_mm)
    ftle_cmap=plt.get_cmap('viridis').copy();ftle_cmap.set_bad('#E5E5E5')
    error_cmap=plt.get_cmap('magma').copy();error_cmap.set_bad('#E5E5E5')
    vmin=min(float(truth[mask].min()),*(float(d['prediction'][mask].min()) for d in arrays.values()))
    vmax=max(float(truth[mask].max()),*(float(d['prediction'][mask].max()) for d in arrays.values()))
    error_max=max(float(abs(d['prediction']-truth)[mask].max()) for d in arrays.values())
    assert error_max>0
    images=[]
    for i in range(6):
        for j in range(2):
            ax=axes[i,j]
            if i==0:
                data=truth if j==0 else mask.astype(float)
                title='Reference FTLE' if j==0 else 'Shared evaluation support'
                values=transform(np.ma.array(data,mask=~mask)) if j==0 else transform(data)
                cmap=ftle_cmap if j==0 else ListedColormap(['#E5E5E5','#667788'])
                limits=(vmin,vmax) if j==0 else (0,1)
            else:
                method,label=methods[i-1];prediction=arrays[method]['prediction']
                data=prediction if j==0 else abs(prediction-truth)
                title=label if j==0 else label+' | absolute error'
                values=transform(np.ma.array(data,mask=~mask));cmap=ftle_cmap if j==0 else error_cmap
                limits=(vmin,vmax) if j==0 else (0,error_max)
            im=ax.imshow(values,origin='lower',extent=extent,interpolation='nearest',aspect='equal',
                         cmap=cmap,vmin=limits[0],vmax=limits[1])
            if (i,j) in ((0,0),(1,1)):images.append(im)
            ax.set_title(title,fontsize=7,loc='left',pad=4)
            ax.set_xticks([extent[0],extent[1]]);ax.set_yticks([extent[2],extent[3]])
            ax.tick_params(labelsize=6,pad=1,length=2)
            if i<5:ax.tick_params(labelbottom=False)
            if i==5:ax.set_xlabel(xlabel,labelpad=2)
            ax.set_ylabel(ylabel,labelpad=2,rotation=0,rotation_mode='anchor',va='center')
            panel_label(ax,chr(97+2*i+j))
    t0,t1=row['integration']['t0'],row['integration']['t1']
    fig.suptitle(f'{FLOW_NAMES[flow]} | {scale}× | t = {t0:.3f} to {t1:.3f} | seed {seed}',fontsize=9,y=1-3/figure_height)
    cax1=fig.add_axes([.15,13/figure_height,.30,2.4/figure_height])
    cax2=fig.add_axes([.64,13/figure_height,.30,2.4/figure_height])
    for image,cax,label in zip(images,[cax1,cax2],['FTLE','Absolute FTLE error']):
        bar=fig.colorbar(image,cax=cax,orientation='horizontal');bar.ax.tick_params(labelsize=6,length=2,pad=1)
        bar.set_label(label,fontsize=7,labelpad=2)
    fig.text(.11,2/figure_height,'Grey: outside the scored region. Identical color scales across methods; residuals are not clipped.',fontsize=6)
    name=f'{flow}_x{scale}'
    export(fig,out/name,list(axes.flat))
    dump(out/(name+'.source.json'),{'flow':flow,'scale':scale,'seed':seed,'file':first,'record':row,
                                  'prediction_sha256':hashes,'transposed_axes':transpose,'physical_bounds':bounds,
                                  'pixels_displayed':int(mask.sum()),'total_grid_pixels':mask.size,
                                  'ftle_color_limits':[vmin,vmax],'absolute_error_color_limits':[0,error_max],
                                  'selection_rule':'First final seed and lexicographically first test slice, fixed before final evaluation.'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('outputs/Other_FTLEP35Fusion_2.1/final'))
    parser.add_argument('--source-root',type=Path,default=Path('outputs/Other_FTLEUpsampling2D_1.2/data'))
    parser.add_argument('--skill-scripts',type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.skill_scripts))
    assert read(args.root/'audit.json')['status']=='PASS'
    summary=read(args.root/'summary.json');lock=read(args.root/'lock.json')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
                         'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False,
                         'axes.linewidth':.65,'legend.frameon':False,'savefig.facecolor':'white'})
    out=args.root/'figures';out.mkdir(parents=True,exist_ok=True)
    flows=list(dict.fromkeys(r['flow'] for r in summary['rows']))
    gains(args.root,out,summary,flows)
    for flow in flows:
        for scale in (4,8):plate(args.root,out,summary,flow,scale,lock['seeds'][0],args.source_root)
    dump(out/'manifest.json',{'summary_sha256':sha(args.root/'summary.json'),'lock_sha256':sha(args.root/'lock.json'),
                              'plot_source_sha256':sha(__file__),'backend':'matplotlib','seed':lock['seeds'][0]})


if __name__=='__main__':main()
