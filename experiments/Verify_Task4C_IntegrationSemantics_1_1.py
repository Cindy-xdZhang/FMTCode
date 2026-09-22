"""Read-only integration diagnosis; never changes datasets or training inputs."""
import json
from pathlib import Path
import hashlib
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from scipy.interpolate import RegularGridInterpolator
from FMT_Utils.Task4C_FixedDataset_2_1 import trace_halves

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/mainExp_Task4C_CouetteDataset_1.1'
OUT = ROOT / 'outputs/Verify_Task4C_IntegrationSemantics_1.1'


def arc(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def main():
    settings = json.loads((ROOT/'config/mainExp_Task4C_V2NewLabel_1.1.json').read_text())['integration']
    constant = []
    for magnitude in (1., 10., 100.):
        grid = vtk.vtkImageData(); grid.SetDimensions(3, 3, 3); grid.SetOrigin(-2, -2, -2); grid.SetSpacing(2, 2, 2)
        values = np.zeros((27, 3)); values[:, 0] = magnitude
        array = numpy_to_vtk(values, deep=True); array.SetName('vorticity'); grid.GetPointData().AddArray(array)
        halves, _, _ = trace_halves(grid, np.zeros((1, 3)), .0002, 50, settings)
        constant.append(dict(magnitude=magnitude, vtk_forward_arc=arc(halves[1][0]), raw_ode_50_step_arc=magnitude*.0002*50))
    axes = [np.load(DATA/f'field_cache/axis{i}.npy') for i in range(3)]
    omega = np.load(DATA/'field_cache/omega.npy', mmap_mode='r')
    interp = RegularGridInterpolator(tuple(axes[::-1]), omega, bounds_error=False, fill_value=np.nan)
    seeds = np.load(DATA/'physical/couette/seeds.npy', mmap_mode='r')
    w = interp(seeds[:, ::-1]); norms = np.linalg.norm(w, axis=1)
    selected = [184465, 27722, 194555, 51812]
    examples = []
    for index in selected:
        seed = np.array(seeds[index], dtype=float)
        def f(x): return interp(x[None, ::-1])[0]
        record = dict(seed_index=index, seed=seed.tolist(), omega=f(seed).tolist(), omega_norm=float(np.linalg.norm(f(seed))))
        variants = {}
        for updates in (49, 50):
            lengths = []
            for sign in (-1, 1):
                points = [seed.copy()]; h = sign*.0002
                for _ in range(updates):
                    x = points[-1]; k1 = f(x); k2 = f(x+h*k1/2); k3 = f(x+h*k2/2); k4 = f(x+h*k3)
                    new = x+h*(k1+2*k2+2*k3+k4)/6
                    if not np.isfinite(new).all() or any(new[d]<axes[d][0] or new[d]>axes[d][-1] for d in range(3)): break
                    points.append(new)
                lengths.append(dict(arc=arc(points), updates=len(points)-1))
            variants[str(updates)] = lengths
        curves = np.load(DATA/'physical/couette/curves.npy', mmap_mode='r')
        record.update(raw_omega_rk4=variants, saved_middle_merged_arc=arc(curves[index, 1]))
        examples.append(record)
    sources = [ROOT/'FMT_Utils/Task4C_FixedDataset_2_1.py', ROOT/'FMT_Utils/Task4B_CrossFlow_3D.py',
        Path('C:/Users/xingdi/sources/optimal-connection/src/flow3d/Discrete3DFlowField.cpp'),
        Path('C:/Users/xingdi/sources/GenericVariationalVortexCore/src/flow3d/Discrete3DFlowField.cpp')]
    result = dict(version='Verify_Task4C_IntegrationSemantics_1.1', vtk_version=vtk.vtkVersion.GetVTKVersion(),
        constant_field_probe=constant, couette_seed_omega_norm_quantiles=dict(zip(['min','p25','median','p75','max'],np.quantile(norms,[0,.25,.5,.75,1]).tolist())),
        examples=examples, limits='Raw RK4 uses the existing Python curl field, not the C++ executable or its derivative/grid implementation. It isolates integration parameter semantics only.',
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
