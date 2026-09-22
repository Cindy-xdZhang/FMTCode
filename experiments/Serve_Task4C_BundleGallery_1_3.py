"""Original bundle gallery plus on-demand, memory-only Couette exploration."""
import argparse
from collections import OrderedDict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading
from urllib.parse import urlparse, parse_qs

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.spatial import cKDTree

from experiments.Inspect_Task4C_Couette_1_1 import ROOT as INPUT, read, sha
from FMT_Utils.Task4B_CrossFlow_3D import finite_difference_curl_zyx
from FMT_Utils.Task4C_FixedDataset_2_1 import trace_halves, cut_and_clean, half_steps
from FMT_Utils.Task4C_HairpinBinary_2_1 import sample_gt
from FMT_Utils.Task4C_Multiscale_4_1 import interpolate_scalar
from FMT_Utils.Task4C_BallQuery_2_2 import ball_tables

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/Verify_Task4C_BundleGallery_1.1/viewer'
GT_VIEW = ROOT/'outputs/Verify_Task4C_GTHeadLeg_2.3/viewer'
VERSION = 'Other_Task4C_BundleGallery_1.3'
LOCK = threading.Lock()
SCENE = None
CACHE = OrderedDict()


def scene():
    global SCENE
    if SCENE is not None:
        return SCENE
    audit = json.loads((ROOT/'outputs/Verify_Task4C_CouetteInputs_1.1/input_audit.json').read_text())
    for name, item in audit['inputs'].items():
        assert sha(INPUT/name) == item['sha256']
    grid = read(INPUT/'couette.vtk'); gt = read(INPUT/'couette_GTs.vtk')
    dims = [0, 0, 0]; grid.GetDimensions(dims); shape = tuple(dims[::-1])
    xyz = vtk_to_numpy(grid.GetPoints().GetData()).reshape(*shape, 3)
    axes = [xyz[0, 0, :, 0].copy(), xyz[0, :, 0, 1].copy(), xyz[:, 0, 0, 2].copy()]
    velocity = vtk_to_numpy(grid.GetPointData().GetArray('velocity')).reshape(*shape, 3)
    omega = finite_difference_curl_zyx(velocity, tuple(axes[::-1]))
    arr = numpy_to_vtk(np.ascontiguousarray(omega.reshape(-1, 3)), deep=True)
    arr.SetName('vorticity'); grid.GetPointData().AddArray(arr)
    locator = vtk.vtkStaticCellLocator(); locator.SetDataSet(gt); locator.BuildLocator()
    centers_file = ROOT/'outputs/Verify_Task4C_GTHeadLeg_2.3/couette/cell_segmentation.npz'
    with np.load(centers_file) as data:
        centers = data['centers'].copy(); instances = data['instance'].copy()
    integration = json.loads((ROOT/'config/mainExp_Task4C_V2NewLabel_1.1.json').read_text())['integration']
    h = min(float(a[-1]-a[0])/(len(a)-1) for a in axes)
    SCENE = dict(grid=grid, gt=gt, axes=axes, velocity=velocity, omega=omega, locator=locator,
                 centers=centers, instances=instances, tree=cKDTree(centers), h=h, integration=integration)
    return SCENE


def preview(instance, scale):
    """No candidate filter or split is implied; these are exploratory GT-neighborhood seeds."""
    key = (instance, scale)
    with LOCK:
        if key in CACHE:
            CACHE.move_to_end(key)
            return CACHE[key]
        s = scene()
        mesh = json.loads((GT_VIEW/f'instances/couette_{instance}.json').read_text(encoding='utf8'))
        bounds = np.asarray(mesh['bounds']).reshape(3, 2)
        rng = np.random.default_rng(96611+instance)
        inside = s['centers'][s['instances'] == instance]
        # Half GT-cell centers and half uniform points in its padded box, solely for inspection.
        center_seeds = inside[rng.choice(len(inside), 256, replace=False)]
        lo = np.maximum(bounds[:, 0]-.15*(bounds[:, 1]-bounds[:, 0]), [a[0]+1e-7 for a in s['axes']])
        hi = np.minimum(bounds[:, 1]+.15*(bounds[:, 1]-bounds[:, 0]), [a[-1]-1e-7 for a in s['axes']])
        seeds = np.vstack([center_seeds, rng.uniform(lo, hi, (256, 3))])
        seeds = seeds[rng.permutation(len(seeds))]
        lengths = np.asarray([.07, .10, .13])*scale
        ds = .001
        steps = half_steps(lengths, ds)
        halves, reasons, termination = trace_halves(s['grid'], seeds, ds, steps[-1], s['integration'])
        keep, curves, raw_lines = [], [], []
        for i, seed in enumerate(seeds):
            back = halves[0].get(i, seed[None]); front = halves[1].get(i, seed[None])
            line, valid, *_ = cut_and_clean(back, front, termination[i], ds, steps, s['integration'])
            if valid.all():
                keep.append(i); curves.append(line)
                raw_lines.append(np.concatenate((back[:0:-1], front)))
        if len(keep) <= 16:
            raise ValueError(f'Only {len(keep)} valid seeds; reduce the length multiplier.')
        seeds = seeds[keep]; curves = np.asarray(curves)
        owners, _ = sample_gt(s['gt'], curves[:, 0].reshape(-1, 3), s['locator'])
        short = (owners.reshape(-1, 32) >= 0).sum(1)
        raw = np.concatenate(raw_lines)
        raw_owner, _ = sample_gt(s['gt'], raw, s['locator'])
        vectors = [np.column_stack([interpolate_scalar(raw, s['axes'], s[name][..., d])[0] for d in range(3)])
                   for name in ('velocity', 'omega')]
        v, w = vectors; norm = (v*v).sum(1)*(w*w).sum(1); dot = (v*w).sum(1)
        head = (w[:, 1] > 0) & (norm > 0) & (2*dot*dot < norm)
        offsets = np.r_[0, np.cumsum([len(line) for line in raw_lines])]
        heads = np.add.reduceat(((raw_owner >= 0) & head).astype(int), offsets[:-1])
        legs = np.add.reduceat(((raw_owner >= 0) & ~head).astype(int), offsets[:-1])
        labels = (short >= 17) & (heads > 0) & (legs > 0)
        seed_owner, _ = sample_gt(s['gt'], seeds, s['locator'])
        assigned = np.where(seed_owner >= 0, seed_owner, s['instances'][s['tree'].query(seeds)[1]])
        tables = ball_tables(seeds, 3*s['h'], workers=1)
        # Independent radius, uniqueness, and label accounting checks on all displayed data.
        for k in (6, 16):
            ids = tables[f'order{k}']
            assert all(len(set(row)) == k for row in ids)
            distances = np.linalg.norm(seeds[ids]-seeds[:, None], axis=2)
            assert np.all(distances <= tables[f'effective_radius{k}'][:, None]+1e-12)
        assert np.all(heads+legs <= np.diff(offsets))
        result = dict(version=VERSION, preview_only=True, split=None, candidate_filter_applied=False,
            instance=instance, scale=scale, half_lengths=lengths.tolist(), ds=ds, h=s['h'],
            requested_seeds=512, valid_seeds=len(seeds), rejected_seeds=512-len(seeds),
            positive=int(labels.sum()), negative=int((~labels).sum()),
            sampling='256 GT cell centers + 256 uniform padded-box seeds; no training/test split or candidate filter',
            label_rule='short32_gt_count >= 17 AND longest_raw_gt_head > 0 AND longest_raw_gt_leg > 0',
            records=[dict(row=i, seed=p.tolist(), label=int(labels[i]), instance=int(assigned[i]),
                seed_gt_instance=int(seed_owner[i]), short_gt_count=int(short[i]), long_head=int(heads[i]),
                long_leg=int(legs[i]), long_raw_points=int(offsets[i+1]-offsets[i]),
                initial_count=int(tables['initial_count'][i])) for i, p in enumerate(seeds)],
            curves=curves.tolist(), neighbors={str(k):dict(ids=tables[f'order{k}'].tolist(),
                effective_radius=tables[f'effective_radius{k}'].tolist(), expanded=tables[f'expanded{k}'].tolist()) for k in (6, 16)})
        CACHE[key] = result
        while len(CACHE) > 2:
            CACHE.popitem(last=False)
        print(json.dumps({k:result[k] for k in ('instance','scale','valid_seeds','positive','negative')}), flush=True)
        return result


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE), **kwargs)

    def send_data(self, content, kind='application/json; charset=utf-8', status=200):
        body = content.encode('utf8') if isinstance(content, str) else json.dumps(content, ensure_ascii=False, allow_nan=False).encode('utf8')
        self.send_response(status); self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == '/api/status':
                return self.send_data(dict(version=VERSION, field_loaded=SCENE is not None, memory_previews=len(CACHE)))
            if url.path == '/api/couette':
                query = parse_qs(url.query); instance = int(query.get('instance', ['0'])[0]); scale = float(query.get('scale', ['1'])[0])
                if instance not in range(6) or scale not in (.5, 1., 2., 4.):
                    raise ValueError('Unsupported instance or length multiplier')
                return self.send_data(preview(instance, scale))
            if url.path.startswith('/api/mesh/'):
                iid = int(url.path.rsplit('/', 1)[-1])
                if iid not in range(6):
                    raise ValueError('Unknown instance')
                data = json.loads((GT_VIEW/f'instances/couette_{iid}.json').read_text(encoding='utf8'))
                return self.send_data(dict(flow='couette', instance=iid, **data['original']))
            if url.path in ('/', '/index.html'):
                html = (BASE/'index.html').read_text(encoding='utf8')
                html = html.replace('</body>', '<script src="/couette_gallery.js"></script></body>')
                return self.send_data(html, 'text/html; charset=utf-8')
            if url.path == '/couette_gallery.js':
                return self.send_data((ROOT/'experiments/templates/task4c_bundle_gallery_1_3.js').read_text(encoding='utf8'), 'text/javascript; charset=utf-8')
            return super().do_GET()
        except (ValueError, AssertionError) as exc:
            return self.send_data(dict(error=str(exc)), status=400)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return self.send_data(dict(error=str(exc)), status=500)


class GalleryServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--port', type=int, default=8768)
    args = parser.parse_args()
    print(f'{VERSION}: http://127.0.0.1:{args.port}/index.html', flush=True)
    GalleryServer(('127.0.0.1', args.port), Handler).serve_forever()
