"""Seven-observer scalar-feature figure, structurally adapted from the existing renderer."""
from pathlib import Path
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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
        ax.set_axis_off()  # No fixed-domain box in a moving coordinate system.
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
           'view_zoom':2.1,'artist_clipping':False,'bounding_box_drawn':False,
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
