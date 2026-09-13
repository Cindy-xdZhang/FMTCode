"""Interactive 3D inspection of downloaded velocity; no synthetic flow data."""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def render(directory):
    out=Path(directory)
    m=json.loads((out/'manifest.json').read_text())
    c=m['config']
    records=sorted(m['frames'],key=lambda f:f['index'])
    if not records:
        raise ValueError('No completed frames to visualize')
    axes=[np.linspace(*c[f'{a}_range'],c[f'{a}_res']) for a in 'xyz']
    ids=[np.unique(np.linspace(0,len(axis)-1,min(20,len(axis))).astype(int)) for axis in axes]
    z,y,x=np.meshgrid(axes[2][ids[2]],axes[1][ids[1]],axes[0][ids[0]],indexing='ij')
    arrays=[]
    for r in records:
        a=np.load(out/r['file'],mmap_mode='r')
        arrays.append(a[np.ix_(ids[2],ids[1],ids[0])])
    speeds=[np.linalg.norm(a,axis=-1) for a in arrays]
    lo=min(float(s.min()) for s in speeds)
    hi=max(float(s.max()) for s in speeds)
    def title(t):
        return (f'JHTDB channel | {c["x_res"]} x {c["y_res"]} x {c["z_res"]} | '
                f'{len(records)} frames | t = {t:.4f}<br>'
                f'<sup>Fixed physical coordinates; displayed on {len(ids[0])} x {len(ids[1])} x {len(ids[2])} points</sup>')
    frames=[]
    for r,a,speed in zip(records,arrays,speeds):
        volume=go.Volume(x=x.ravel(),y=y.ravel(),z=z.ravel(),value=speed.ravel(),
                         isomin=lo,isomax=hi,cmin=lo,cmax=hi,opacity=.12,surface_count=12,
                         colorscale='Viridis',colorbar=dict(title='Speed |velocity|'),
                         caps=dict(x_show=False,y_show=False,z_show=False),
                         name='Speed',hovertemplate='x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<br>speed=%{value:.4f}<extra></extra>')
        skip=(slice(None,None,4),)*3
        cone=go.Cone(x=x[skip].ravel(),y=y[skip].ravel(),z=z[skip].ravel(),
                     u=a[skip][...,0].ravel(),v=a[skip][...,1].ravel(),w=a[skip][...,2].ravel(),
                     sizemode='absolute',sizeref=.022,anchor='tail',showscale=False,
                     colorscale=[[0,'#31465a'],[1,'#31465a']],name='Velocity direction',
                     hovertemplate='u=%{u:.4f}<br>v=%{v:.4f}<br>w=%{w:.4f}<extra></extra>')
        frames.append(go.Frame(name=str(r['index']),data=[volume,cone],layout=go.Layout(title_text=title(r['time']))))
    fig=go.Figure(data=frames[0].data,frames=frames)
    steps=[dict(label=f'{r["index"]+1}',method='animate',args=[[str(r['index'])],dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))]) for r in records]
    fig.update_layout(title=dict(text=title(records[0]['time']),font_size=20),
                      scene=dict(xaxis_title='x',yaxis_title='y',zaxis_title='z',aspectmode='data',
                                 camera=dict(eye=dict(x=1.6,y=1.4,z=1.1))),
                      height=820,margin=dict(l=30,r=30,b=90,t=100),template='plotly_white',
                      sliders=[dict(active=0,currentvalue=dict(prefix='Frame = '),steps=steps,pad=dict(t=30))],
                      updatemenus=[dict(type='buttons',direction='left',x=0,y=0,buttons=[
                          dict(label='Play',method='animate',args=[None,dict(frame=dict(duration=250,redraw=True),transition=dict(duration=0),fromcurrent=True)]),
                          dict(label='Pause',method='animate',args=[[None],dict(mode='immediate',frame=dict(duration=0,redraw=False))])])])
    fig.write_html(out/'channel_3d.html',include_plotlyjs=True,auto_open=False,auto_play=False)
    print(out/'channel_3d.html')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',nargs='?',default='outputs/Verify_JHTDB_ChannelDownload_1.1')
    render(p.parse_args().directory)
