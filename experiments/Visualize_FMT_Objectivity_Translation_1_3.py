"""Four vertically stacked observed-pathline classification panels.

Compare 0%, 25%, 50%, 100% of a supplied translating observer's VELOCITY.
This is a test of objectivity, not an assumption that predictions are invariant.
The default CLI adapter uses the frozen Task1 4.1 recipe; render() and
evaluate_observers() also support another already-fitted classifier callback.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tmp/task123_plotdeps'),str(ROOT)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import RegularGridInterpolator
from FMT_Utils.TranslationObserver_3D import TranslationObserver,evaluate_observers


def render(paths,labels,original_indices,audit,output_dir,title,medium='paper',pdf_collision_audit=True):
    """Color entire center trajectories, with fixed camera and union bounds."""
    skill=Path(os.environ.get('NATURE_FIGURE_SKILL_ROOT',Path.home()/'.codex/skills/nature-figure'))
    sys.path.insert(0,str(skill/'scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    paper=medium=='paper';font=7 if paper else 13
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':font,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.2,10.2) if paper else (10.8,15.3),facecolor='white')
    center=paths[:,:,0]
    flat=center.reshape(-1,3);low=flat.min(axis=0);high=flat.max(axis=0)
    span=high-low;pad=np.maximum(span*.035,max(span.max(),1.)*.005)
    low-=pad;high+=pad
    axes=[]
    for i,(alpha,trajectories,y) in enumerate(zip(audit['levels'],center,labels)):
        bottom=.735-i*.225
        ax=fig.add_axes([.06,bottom,.87,.19],projection='3d');axes.append(ax)
        ax.set_proj_type('ortho');ax.view_init(elev=22,azim=-62)
        ax.set_box_aspect(high-low,zoom=2.1)
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]))
        ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([]);ax.grid(False)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False
        for value,color in [(False,'#3776A6'),(True,'#B63B36')]:
            collection=Line3DCollection(trajectories[y==value],colors=color,
                                        linewidths=(.55 if value else .20)*(1 if paper else 1.5),
                                        alpha=.95 if value else .12)
            collection.set_rasterized(True);ax.add_collection3d(collection)
        fig.text(.045,bottom+.188,'abcd'[i],fontweight='bold',fontsize=font+2)
        heading='Original frame (0%)' if i==0 else f'{alpha:.0%} target observer velocity'
        fig.text(.16,bottom+.191,heading,fontsize=font+1)
        changes=audit['changed_labels_vs_original'][i]
        fig.text(.5,bottom-.002,f'Vortex: {y.sum():,} / {len(y):,}    |    Changed labels: {changes:,}',
                 fontsize=font,ha='center')
    fig.text(.5,.971,title,ha='center',fontsize=font+2,fontweight='bold')
    fig.legend(handles=[Line2D([],[],color=c,lw=2,label=l) for c,l in
                        [('#B63B36','Vortex pathline'),('#3776A6','Non-vortex pathline')]],
               loc='lower center',bbox_to_anchor=(.5,.018),ncol=2,frameon=False,fontsize=font)
    fig.text(.5,.006,'Same material seeds, classifier, camera and spatial scale',ha='center',fontsize=font)
    stem=output_dir/medium
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=list('abcd'),
                  column_groups=[list('abcd')],strict=True,tolerance_pt=1.5,
                  json_out=str(stem)+'.alignment.json')
    for suffix in ['.pdf','.svg','.png']:
        fig.savefig(stem.with_suffix(suffix),dpi=400 if paper else 220)
    if paper:fig.savefig(stem.with_suffix('.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    audit={**audit,'display':'All surviving center paths; no point-only labels or prediction-based selection',
           'shared_bounds':[low.tolist(),high.tolist()],'camera':[22,-62],'medium':medium,
           'minimum_font_pt':font,'figure_size_inches':[7.2,10.2] if paper else [10.8,15.3],
           'view_zoom':2.1,'line_opacity':{'vortex':.95,'non_vortex':.12},
           'style_reason':'Keep every trajectory visible while reducing dense non-vortex occlusion'}
    stem.with_suffix('.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    np.savez_compressed(output_dir/'observed_pathlines.npz',pathlines=paths,
                       predictions=labels,original_indices=original_indices,
                       sample_times=np.asarray(audit['sample_times']))
    # Audit the exact exported PDF, after final layout and rasterization.
    import audit_pdf_text
    fonts=audit_pdf_text.audit_pdf(stem.with_suffix('.pdf').read_bytes(),5)
    if pdf_collision_audit:
        import audit_figure_collisions
        collision=audit_figure_collisions.audit_pdf(stem.with_suffix('.pdf'))
    else:
        collision={'status':'PENDING_LOCAL_AUDIT','reason':'Remote PyMuPDF unavailable; audit downloaded final PDF before delivery'}
    Path(str(stem)+'.font-audit.json').write_text(json.dumps(fonts,indent=2))
    Path(str(stem)+'.collision-audit.json').write_text(json.dumps(collision,indent=2))
    if fonts['below_minimum_count'] or (pdf_collision_audit and collision['summary']['fail']):
        raise RuntimeError('Final PDF failed font/collision audit; review the saved reports.')
    return stem.with_suffix('.png')


def task1_inputs(args):
    """Fit once on original training/calibration; never fit per observer."""
    import torch
    import yaml
    from FMT_Utils.Task12Data_3D import load_cache_records,stack_features,stack_reference,feature_matrix
    from FMT_Utils.Task12Evaluation_3D import fit_kmeans_transform,calibrate_vortex_cluster
    from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
    from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d
    spec=yaml.safe_load(args.protocol_config.read_text())
    selected_path=args.selection or Path(spec['output_dir'])/'selected_config.json'
    selected=json.loads(selected_path.read_text())
    if selected.get('confirmation_opened',True):
        raise ValueError('Task1 selection must be frozen before confirmation was opened.')
    if selected['config_sha256']!=hashlib.sha256(args.protocol_config.read_bytes()).hexdigest():
        raise ValueError('Frozen selection/config hash mismatch.')
    source=next(s for s in spec['sources'].values() if args.dataset in s['datasets'])
    records=load_cache_records(Path(source['development_cache'])/args.dataset,10)
    train=[records[i] for i in spec['splits']['final_train']]
    calibration=[records[i] for i in spec['splits']['cluster_calibration']]
    choice=selected['fmt'];dim=choice['pca_dim']
    device=torch.device(args.device)
    model=fit_kmeans_transform(stack_features(train,choice['feature'],device),
                               None if dim in (None,'none') else int(dim),7080,spec['kmeans_n_init'])
    vortex=calibrate_vortex_cluster(stack_reference(calibration),
                          model.predict(stack_features(calibration,choice['feature'],device)))
    def classifier(primitives):
        xyz=np.asarray(primitives,dtype=np.float32)
        raw=(xyz-xyz[:,:1,:1]).reshape(len(xyz),-1)
        fmt=pathline_dft_features_3d(torch.from_numpy(xyz).to(device),num_freq=6,
               neighbor_weight=1.,neighbor_scale=1.,neighbor_pool='sort',mode='gram',include_chirality=True)
        feature=feature_matrix({'raw':raw,'fmt':fmt,'features':{}},choice['feature'],device)
        return model.predict(feature)==vortex,feature
    from FMT_Utils.CylinderTimePolicy import select_from_files
    cache_dir=Path(source['confirmation_cache'])/args.dataset
    ordinal,time_audit=select_from_files(cache_dir,args.dataset,args.time_policy)
    print(json.dumps(time_audit),flush=True)
    record=load_cache_records(cache_dir,4,[ordinal])[0]
    meta=record['metadata']
    path=args.source_file or Path(meta['source_path'])
    if not path.exists():
        basename=str(meta['source_path']).replace('\\','/').rsplit('/',1)[-1]
        path=Path('/home/zhanx0o/DeepVortex/FLowDataFolder')/basename
    if not path.exists():raise FileNotFoundError('Supply --source-file for the original NetCDF field.')
    field,_=load_netcdf_window_3d(path,meta['source_start_index'],meta['frame_count'],96)
    values=field.field
    low=np.asarray(field.domainMinBoundary);high=np.asarray(field.domainMaxBoundary)
    axes=[np.linspace(low[j],high[j],values.shape[3-j]) for j in range(3)]
    t0=float(meta['source_time']);dt=float(field.timeInterval)*.25
    field_times=t0+np.arange(len(values))*float(field.timeInterval)
    interpolator=RegularGridInterpolator((field_times,*axes[::-1]),values,bounds_error=False,fill_value=np.nan)
    def lab_velocity(points,t):
        return interpolator(np.column_stack((np.full(len(points),t),points[:,[2,1,0]])))
    with np.load(record['path']) as cache:
        seeds=cache['seeds']
    # Cached raw coordinates subtract only the original center seed, not a scale.
    local=record['raw'].reshape(-1,7,32,3)
    initial=local[:,:,0].astype(np.float64)+seeds[:,None,:]
    indices=np.linspace(0,48,32).round().astype(int)
    times=t0+indices*dt
    identity={'experiment':'Other_FMTObjectivityTranslation_1.3','dataset':args.dataset,
              'task':'Task1','protocol':'mainExp_Task1_3D_4.1','seed':7080,'confirmation_ordinal':ordinal,
              'fmt_recipe':choice,'selection_sha256':hashlib.sha256(selected_path.read_bytes()).hexdigest(),
              'cache_sha256':hashlib.sha256(record['path'].read_bytes()).hexdigest(),
              'source_path':str(path),'source_time':t0,'new_training_per_observer':False,
              'scope':'Pure translation only; no assertion of rotation or arbitrary-observer invariance'}
    if args.global_mean:
        from FMT_Utils.GlobalMeanObserver_3D import source_mean_schedule
        mean_times, means, mean_info = source_mean_schedule(path, meta['source_start_index'], meta['frame_count'])
        identity.update(mean_info)
        identity['target_mean_times'] = mean_times.tolist()
        identity['target_mean_velocities'] = means.tolist()
        identity['mean_velocity_range'] = [means.min(axis=0).tolist(), means.max(axis=0).tolist()]
    identity.update(time_audit)
    identity['source_start_index'] = meta['source_start_index']
    identity['loaded_field_shape_tzyxc'] = list(values.shape)
    identity['source_file_size_bytes'] = path.stat().st_size
    identity['cached_reference'] = record['reference'].tolist()
    return lab_velocity,initial,times,classifier,dt,identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    p.add_argument('--protocol-config',type=Path,default=ROOT/'config/mainExp_Task1_3D_4.1_uniform.yaml')
    p.add_argument('--selection',type=Path)
    p.add_argument('--source-file',type=Path)
    p.add_argument('--time-policy',type=Path,default=ROOT/'config/cylinder_time_policy.json')
    p.add_argument('--defer-pdf-collision-audit',action='store_true',help='Audit final downloaded PDFs locally before delivery')
    observer=p.add_mutually_exclusive_group(required=True)
    observer.add_argument('--global-mean', action='store_true', help='Full-source spatial volume mean at every time')
    observer.add_argument('--target-velocity',type=float,nargs=3,metavar=('VX','VY','VZ'))
    observer.add_argument('--observer-json',type=Path,help='Physical times and target lab-frame velocity samples; optional time_origin=relative_to_seed')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--substeps',type=int,default=2)
    p.add_argument('--correspondence-tolerance',type=float,default=1e-5)
    p.add_argument('--media',nargs='+',choices=['paper','slides'],default=['paper','slides'])
    args=p.parse_args()
    if args.substeps<1:p.error('--substeps must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new output directory; existing observer results are not overwritten.')
    v,initial,times,classifier,dt,identity=task1_inputs(args)
    if args.global_mean:
        schedule=TranslationObserver(identity['target_mean_times'],identity['target_mean_velocities'],times[0])
    elif args.target_velocity is not None:
        schedule=TranslationObserver([times[0],times[-1]],[args.target_velocity]*2,times[0])
    else:
        s=json.loads(args.observer_json.read_text())
        identity['observer_schedule_sha256']=hashlib.sha256(args.observer_json.read_bytes()).hexdigest()
        identity['observer_schedule_provenance']={k:v for k,v in s.items() if k not in ['times','velocities']}
        if 'window_file_sha256' in s:
            actual=hashlib.sha256(Path(identity['source_path']).read_bytes()).hexdigest()
            if actual!=s['window_file_sha256']:
                raise ValueError('Source window does not match original-source extraction manifest.')
            identity['source_window_sha256']=actual
        knots=np.asarray(s['times'],dtype=float)
        origin=s.get('time_origin','physical')
        if origin=='relative_to_seed':knots=knots+times[0]
        elif origin!='physical':raise ValueError('Unknown observer time_origin.')
        schedule=TranslationObserver(knots,s['velocities'],times[0])
    paths,labels,ids,audit=evaluate_observers(v,initial,times,classifier,schedule,
                         dt/args.substeps,args.correspondence_tolerance)
    reference=np.asarray(identity.pop('cached_reference'),dtype=bool)[ids]
    from sklearn.metrics import f1_score
    audit['ivd_reference_positive_count']=int(reference.sum())
    audit['f1_vs_fixed_ivd_reference']=[float(f1_score(reference,y,zero_division=0)) for y in labels]
    audit['vortex_counts']=[int(y.sum()) for y in labels]
    audit.update(identity)
    audit['observer_displacement_at_end']=schedule.displacement(times[-1]).tolist()
    audit['mean_velocity_over_path_window']=(schedule.displacement(times[-1])/(times[-1]-times[0])).tolist()
    audit['job_id']=os.environ.get('SLURM_JOB_ID')
    audit['node']=os.environ.get('SLURMD_NODENAME')
    print(json.dumps({k:audit[k] for k in ['mean_velocity_over_path_window','observer_displacement_at_end','vortex_counts','changed_labels_vs_original','f1_vs_fixed_ivd_reference','trajectory_correspondence_max_error','common_primitives']},indent=2),flush=True)
    audit['render_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for medium in args.media:
        image=render(paths,labels,ids,audit,args.output_dir,
                     f"{dict(cylinder3d='Re160',halfcylinderRe640='Re640',halfcylinderRe6400='Re6400')[args.dataset]} 3D | Mean-flow observer | t0 = {times[0]:g}",medium,
                     pdf_collision_audit=not args.defer_pdf_collision_audit)
        print(image)
    print('Changed labels:',audit['changed_labels_vs_original'])


if __name__=='__main__':main()
