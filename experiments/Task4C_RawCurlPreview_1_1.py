"""In-memory raw-curl RK4 preview on frozen Couette seeds; no dataset writes."""
from functools import lru_cache
from pathlib import Path
from threading import Lock
import numpy as np
from scipy.interpolate import RegularGridInterpolator

DATA = Path(__file__).resolve().parents[1]/'outputs/mainExp_Task4C_CouetteDataset_1.1'
_lock = Lock()
_scene = None


def scene():
    global _scene
    with _lock:
        if _scene is None:
            axes = [np.load(DATA/f'field_cache/axis{d}.npy') for d in range(3)]
            omega = np.load(DATA/'field_cache/omega.npy', mmap_mode='r')
            _scene = (axes, RegularGridInterpolator(tuple(axes[::-1]), omega, bounds_error=False, fill_value=np.nan),
                      np.load(DATA/'physical/couette/seeds.npy', mmap_mode='r'))
    return _scene


def integrate(seeds, steps, dt, field, low, high):
    halves = []
    for sign in (-1, 1):
        x = np.array(seeds, dtype=np.float64); live = np.ones(len(x), bool)
        paths = [[p.tolist()] for p in x]; reasons = ['completed']*len(x)
        for _ in range(steps):
            ids = np.flatnonzero(live)
            if not len(ids): break
            p = x[ids]; h = sign*dt
            k1 = field(p); k2 = field(p+h*k1/2); k3 = field(p+h*k2/2); k4 = field(p+h*k3)
            q = p+h*(k1+2*k2+2*k3+k4)/6
            finite = np.isfinite(q).all(1)
            inside = finite & (q>=low).all(1) & (q<=high).all(1)
            moving = np.linalg.norm(q-p,axis=1)>1e-14
            for j, index in enumerate(ids):
                if inside[j] and moving[j]:
                    x[index] = q[j]; paths[index].append(q[j].tolist())
                else:
                    live[index] = False
                    reasons[index] = 'boundary_or_rk_stage_outside' if not inside[j] else 'stagnation'
        halves.append((paths, reasons))
    curves=[];stats=[]
    for i in range(len(seeds)):
        back, front = halves[0][0][i], halves[1][0][i]
        curves.append(back[:0:-1]+front)
        arcs = [float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()) if len(p)>1 else 0. for p in (back,front)]
        stats.append(dict(half_arcs=arcs, total_arc=sum(arcs), half_updates=[len(back)-1,len(front)-1], termination=[halves[d][1][i] for d in (0,1)]))
    return curves, stats


@lru_cache(maxsize=128)
def preview(ids, steps):
    if steps not in (35,50,65) or not 1<=len(ids)<=17: raise ValueError('Unsupported preview size')
    axes, interp, seeds = scene()
    if any(i<0 or i>=len(seeds) for i in ids): raise ValueError('Invalid seed ID')
    points = np.asarray(seeds[list(ids)])
    curves, stats = integrate(points, steps, .0002, lambda p:interp(p[:,::-1]), np.array([a[0] for a in axes]),np.array([a[-1] for a in axes]))
    return dict(flow='couette', mode='raw_curl_rk4', field='existing native-coordinate finite-difference curl',
                dt=.0002, requested_updates_per_direction=steps, parameter_interval_per_direction=steps*.0002,
                seed_ids=list(ids), seeds=points.tolist(), curves=curves, stats=stats,
                labels_recomputed=False, saved_as_dataset=False,
                note='Raw-vector preview, not an exact reproduction of the C++ executable. Each step means one RK4 update; the seed is extra.')
