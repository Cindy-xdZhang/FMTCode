"""Inspect velocity-derived Q isosurfaces at the first downloaded time only."""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from FLowUtils.flowDatasetUtils.JHTDB_VTK import read_frame, geometry
from experiments.Download_JHTDB_Channel import write_json, sha
from experiments.Download_JHTDB_VTK import axes_for


def q_criterion(values, axes):
    """Q = -trace(grad(u)^2)/2 on physical xyz; centered second-order differences."""
    x, y, z = axes
    gradients = [np.gradient(values[..., i].astype(np.float64), z, y, x, edge_order=2)
                 for i in range(3)]
    # gradients[component][axis] uses derivative order z,y,x.
    result = np.zeros(values.shape[:3], dtype=np.float64)
    for i in range(3):
        for j in range(3):
            result -= 0.5 * gradients[i][2-j] * gradients[j][2-i]
    return result


def render(directory):
    out = Path(directory)
    manifest = json.loads((out / 'manifest.json').read_text())
    config = manifest['config']
    first = min(manifest['frames'], key=lambda r: r['index'])
    assert first['index'] == 0
    axes = axes_for(config)
    velocity, _ = read_frame(out / first['file'])
    q = q_criterion(velocity, axes)
    # Exclude two outer samples; finite-difference and cropped-surface boundaries remain explicit.
    interior = q[2:-2, 2:-2, 2:-2]
    positive = interior[interior > 0]
    if not positive.size:
        raise ValueError('No positive Q values in this frame')
    level = float(np.percentile(positive, 90))
    grid = geometry([a[2:-2] for a in axes])
    scalar = numpy_to_vtk(np.ascontiguousarray(interior).ravel(), deep=True)
    scalar.SetName('Q')
    grid.GetPointData().SetScalars(scalar)
    contour = vtk.vtkContourFilter()
    contour.SetInputData(grid)
    contour.SetValue(0, level)
    contour.Update()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(contour.GetOutputPort())
    triangles.Update()
    mesh = triangles.GetOutput()
    xyz = vtk_to_numpy(mesh.GetPoints().GetData())
    faces = vtk_to_numpy(mesh.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    fig = go.Figure(go.Mesh3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                              i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                              color='#3288bd', opacity=0.9, flatshading=False,
                              lighting=dict(ambient=.5, diffuse=.8, specular=.25),
                              hovertemplate='x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra>Q isosurface</extra>'))
    name = config['dataset_name']
    fig.update_layout(title=dict(text=f'{name} | velocity-derived Q structure | t={first["time"]:.4f}<br>'
                                 f'<sup>Q={level:.4g}; 90th percentile of positive Q, display only; 128³ velocity grid</sup>'),
                      scene=dict(xaxis=dict(title='x', range=config['x_range']),
                                 yaxis=dict(title='y', range=config['y_range']),
                                 zaxis=dict(title='z', range=config['z_range']), aspectmode='data',
                                 camera=dict(eye=dict(x=1.6, y=1.4, z=1.1))),
                      height=800, margin=dict(l=20, r=20, t=95, b=45), template='plotly_white',
                      annotations=[dict(text='Q = (rotation-rate norm² − strain-rate norm²)/2. Numerical diagnostic; not a research label.',
                                        x=0, y=0, xref='paper', yref='paper', showarrow=False, yshift=-30)])
    fig.write_html(out / 'structure_3d.html', include_plotlyjs=True, auto_open=False)
    report = {'frame': first['file'], 'time': first['time'], 'frame_sha256': sha(out / first['file']),
              'method': 'Q=-trace(grad(u)^2)/2; physical-coordinate second-order finite differences',
              'excluded_boundary_samples': 2, 'positive_Q_display_percentile': 90,
              'Q_isovalue': level, 'vertices': len(xyz), 'triangles': len(faces),
              'purpose': 'Visualization only; does not define or change research labels',
              'source_sha256': sha(Path(__file__))}
    write_json(out / 'structure_preview.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    render(parser.parse_args().directory)
