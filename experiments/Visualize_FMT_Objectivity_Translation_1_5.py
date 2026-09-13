"""Seven vertically stacked observed-pathline classification panels.

Compare seven equally spaced fractions from 0% to 100% of a supplied translating observer's VELOCITY.
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
from FMT_Utils.TranslationObserver_3D import TranslationObserver
from FMT_Utils.SevenTranslationObservers_3D import evaluate_observers


def render(paths,labels,original_indices,audit,output_dir,title,medium='paper',pdf_collision_audit=True):
    """Color entire center trajectories, with fixed camera and union bounds."""
    skill=Path(os.environ.get('NATURE_FIGURE_SKILL_ROOT',Path.home()/'.codex/skills/nature-figure'))
    sys.path.insert(0,str(skill/'scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    paper=medium=='paper';font=7 if paper else 13
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':font,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.2,17.085) if paper else (10.8,25.6275),facecolor='white')
    center=paths[:,:,0]
    flat=center.reshape(-1,3);low=flat.min(axis=0);high=flat.max(axis=0)
    span=high-low;pad=np.maximum(span*.035,max(span.max(),1.)*.005)
    low-=pad;high+=pad
    axes=[]
    for i,(alpha,trajectories,y) in enumerate(zip(audit['levels'],center,labels)):
        bottom=(.735-i*.225+3*.225)/1.675
        ax=fig.add_axes([.06,bottom,.87,.19/1.675],projection='3d');axes.append(ax)
        ax.set_proj_type('ortho');ax.view_init(elev=22,azim=-62)
        ax.set_box_aspect(high-low,zoom=2.1)
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]))
        ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([]);ax.grid(False)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False
        for value,color in [(False,'#3776A6'),(True,'#B63B36')]:
            collection=Line3DCollection(trajectories[y==value],colors=color,
                                        linewidths=(.55 if value else .20)*(1 if paper else 1.5),
                                        alpha=.95 if value else .12)
            collection.set_rasterized(True)
            # The enlarged 3D projection extends beyond Matplotlib's square axes
            # patch. All paths are already inside the validated physical bounds.
            collection.set_clip_on(False)
            ax.add_collection3d(collection)
        fig.text(.045,bottom+.188/1.675,'abcdefg'[i],fontweight='bold',fontsize=font+2)
        heading='Original frame (0%)' if i==0 else f'{100*alpha:.1f}% target observer velocity'
        fig.text(.16,bottom+.191/1.675,heading,fontsize=font+1)
        changes=audit['changed_labels_vs_original'][i]
        fig.text(.5,bottom-.002/1.675,f'Vortex: {y.sum():,} / {len(y):,}    |    Changed labels: {changes:,}',
                 fontsize=font,ha='center')
    fig.text(.5,1-.029/1.675,title,ha='center',fontsize=font+2,fontweight='bold')
    fig.legend(handles=[Line2D([],[],color=c,lw=2,label=l) for c,l in
                        [('#B63B36','Vortex pathline'),('#3776A6','Non-vortex pathline')]],
               loc='lower center',bbox_to_anchor=(.5,.018/1.675),ncol=2,frameon=False,fontsize=font)
    fig.text(.5,.006/1.675,'Labels: first 48 steps; displayed paths: 96 steps (2x duration)' ,ha='center',fontsize=font)
    stem=output_dir/medium
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=list('abcdefg'),
                  column_groups=[list('abcdefg')],strict=True,tolerance_pt=1.5,
                  json_out=str(stem)+'.alignment.json')
    for suffix in ['.pdf','.svg','.png']:
        fig.savefig(stem.with_suffix(suffix),dpi=400 if paper else 220)
    if paper:fig.savefig(stem.with_suffix('.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    audit={**audit,'display':'All surviving center paths; no point-only labels or prediction-based selection',
           'shared_bounds':[low.tolist(),high.tolist()],'camera':[22,-62],'medium':medium,
           'minimum_font_pt':font,'figure_size_inches':[7.2,17.085] if paper else [10.8,25.6275],
           'view_zoom':2.1,'artist_clipping':False,
           'line_opacity':{'vortex':.95,'non_vortex':.12},
           'style_reason':'Keep every trajectory visible while reducing dense non-vortex occlusion'}
    stem.with_suffix('.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
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
    field,_=load_netcdf_window_3d(path,meta['source_start_index'],26,96)
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
    identity={'experiment':'Other_FMTObjectivityTranslation_1.5','dataset':args.dataset,
              'task':'Task1','protocol':'mainExp_Task1_3D_4.1','seed':7080,'confirmation_ordinal':ordinal,
              'fmt_recipe':choice,'selection_sha256':hashlib.sha256(selected_path.read_bytes()).hexdigest(),
              'cache_sha256':hashlib.sha256(record['path'].read_bytes()).hexdigest(),
              'source_path':str(path),'source_time':t0,'new_training_per_observer':False,
              'scope':'Pure translation only; no assertion of rotation or arbitrary-observer invariance'}
    if args.global_mean:
        from FMT_Utils.GlobalMeanObserver_3D import source_mean_schedule
        mean_times, means, mean_info = source_mean_schedule(path, meta['source_start_index'],26)
        identity.update(mean_info)
        identity['target_mean_times'] = mean_times.tolist()
        identity['target_mean_velocities'] = means.tolist()
        identity['mean_velocity_range'] = [means.min(axis=0).tolist(), means.max(axis=0).tolist()]
    identity.update(time_audit)
    identity['source_start_index'] = meta['source_start_index']
    identity['loaded_field_shape_tzyxc'] = list(values.shape)
    identity['source_file_size_bytes'] = path.stat().st_size
    identity['primitive_offset'] = float(meta['primitive_offset'])
    return lab_velocity,initial,times,classifier,dt,identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    p.add_argument('--protocol-config',type=Path,default=ROOT/'config/mainExp_Task1_3D_4.1_uniform.yaml')
    p.add_argument('--selection',type=Path)
    p.add_argument('--time-policy',type=Path,default=ROOT/'config/cylinder_time_policy.json')
    p.add_argument('--previous-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new output directory.')
    previous_file=args.previous_dir/'paper.json'
    previous=json.loads(previous_file.read_text())
    args.source_file=Path(previous['source_path']);args.global_mean=False
    v,unused,short_times,classifier,dt,identity=task1_inputs(args)
    assert identity['cache_sha256']==previous['cache_sha256']
    assert identity['selection_sha256']==previous['selection_sha256']
    assert identity['source_time']==previous['source_time']
    data_file=args.previous_dir/'observed_pathlines.npz'
    with np.load(data_file) as saved:
        old_paths=saved['pathlines']
        initial=old_paths[0,:,:,0,:].copy()
        old_labels=saved['predictions'].copy()
        times=saved['sample_times'].copy()
    assert len(initial)==previous['requested_display_count'] and len(times)==97
    schedule=TranslationObserver(previous['observer_velocity_times'],previous['observer_velocity_samples'],times[0])
    indices=np.asarray(previous['classification_sample_indices'])
    paths,labels,ids,fresh=evaluate_observers(v,initial,times,lambda p:classifier(p[:,:,indices]),schedule,previous['max_step'],previous['correspondence_tolerance'])
    assert len(ids)==len(initial), 'No cohort changes permitted.'
    checks=[]
    for new_i,old_i in [(0,0),(3,2),(6,3)]:
        assert np.array_equal(labels[new_i],old_labels[old_i]), 'Shared-level labels changed.'
        error=float(np.max(np.abs(paths[new_i]-old_paths[old_i])))
        assert error<1e-10, 'Shared-level trajectories changed.'
        checks.append({'alpha':fresh['levels'][new_i],'labels_identical':True,'max_coordinate_difference':error})
    del old_paths
    audit={**previous,**identity,**fresh,'previous_experiment':'Other_FMTObjectivityTranslation_1.4',
           'cohort_reused_exactly':True,'shared_level_regression':checks,
           'previous_audit_sha256':hashlib.sha256(previous_file.read_bytes()).hexdigest(),
           'previous_arrays_sha256':hashlib.sha256(data_file.read_bytes()).hexdigest(),
           'vortex_counts':[int(y.sum()) for y in labels],
           'job_id':os.environ.get('SLURM_JOB_ID'),'node':os.environ.get('SLURMD_NODENAME'),
           'render_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output_dir.mkdir(parents=True)
    # Scientific arrays stay on Ibex; save once independently of export media.
    np.savez_compressed(args.output_dir/'observed_pathlines.npz',pathlines=paths,predictions=labels,original_indices=ids,sample_times=times)
    print(json.dumps({k:audit[k] for k in ['levels','common_primitives','vortex_counts','changed_labels_vs_original','trajectory_correspondence_max_error','shared_level_regression']},indent=2),flush=True)
    for medium in ['paper','slides']:
        title=f"{dict(cylinder3d='Re160',halfcylinderRe640='Re640',halfcylinderRe6400='Re6400')[args.dataset]} 3D | Mean-flow observer | t0 = {times[0]:g}"
        print(render(paths,labels,ids,audit,args.output_dir/'final',title,medium,pdf_collision_audit=False),flush=True)

if __name__=='__main__':main()
