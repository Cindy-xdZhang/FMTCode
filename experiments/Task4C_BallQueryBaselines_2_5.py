"""Conv3D16, BiLSTM and PointNet on the exact shared 3h ball6/ball16 bundles."""
import argparse
import json
from pathlib import Path
import time
from types import FunctionType, SimpleNamespace
import numpy as np
import torch
from experiments import Task4C_FixedDatasetBaselines_2_1 as frozen
from experiments import Task4C_BallQuery_2_5 as fmt
from FMT_Utils import Task4C_BallQuery_2_5 as ball

CONFIGS = {k: f'config/Ablation_Task4C_BallQueryBaselines{k}_2.5.json' for k in (6, 16)}
FAMILIES = ('conv', 'bilstm', 'pointnet')
FILES = tuple(dict.fromkeys(frozen.FILES+fmt.FILES+tuple(CONFIGS.values())))
sha, write = frozen.sha, frozen.write


def identity(config):
    return FunctionType(frozen.identity.__code__, dict(frozen.__dict__, FILES=FILES))(config)


def load_spec(config, family):
    definition = json.loads(Path(config).read_text())
    assert definition['seeds'] == [96611] and definition['neighbor_count'] in (6, 16)
    assert tuple(definition['families']) == FAMILIES
    spec = frozen.load_spec(config, family)
    spec['neighbor_count'] = definition['neighbor_count']
    return spec


def bundle_function(k):
    return FunctionType(frozen.bundle_rows.__code__, dict(frozen.__dict__, LINES=k+1))


def invoke(name, spec, config, *args):
    fn = getattr(frozen, name)
    return FunctionType(fn.__code__, dict(frozen.__dict__, identity=identity, LINES=spec['neighbor_count']+1),
                        argdefs=fn.__defaults__)(spec, config, *args)


def shared_tables(config):
    definition = json.loads(Path(config).read_text()); spec = json.loads(Path(definition['fmt_config']).read_text())
    assert spec['neighbors'] == definition['neighbors'] and spec['source_audit_sha256'] == definition['source_audit_sha256']
    report = json.loads((Path(spec['output'])/'source_verification.json').read_text())
    assert report['complete'] and report['identity'] == fmt.identity(definition['fmt_config'])
    for name, entry in report['neighbor_tables'].items():
        assert sha(Path(spec['output'])/'neighbors'/entry['file']) == entry['sha256']
    return spec, report['neighbor_tables']


def export(config, index):
    spec = load_spec(config, 'conv'); k = spec['neighbor_count']; shared, tables = shared_tables(config)
    adapter = SimpleNamespace(**dict(frozen.v2.__dict__, build_neighbor_tables=lambda *_: tables,
        load_neighbors=lambda _, name: ball.load_neighbors(dict(output=shared['output'], candidate=dict(neighbors=k)), name)))
    FunctionType(frozen.export.__code__, dict(frozen.__dict__, identity=identity, load_spec=load_spec,
        bundle_rows=bundle_function(k), v2=adapter))(config, index)
    path = Path(spec['bundles'])/'physical'/spec['flows'][index]['name']/'preparation.json'
    report = json.loads(path.read_text()); report.update(neighbor_count=k, shared_neighbors_root=shared['output'],
        cluster=f'original sample center + {k} neighbors; initial radius 3h; expand to kth distance if insufficient; full-ball FPS if excess; same curve length')
    write(path, report)


def model_call(spec, config, phase, *args):
    module = frozen.MODULES[spec['family']]; fn = getattr(module, phase)
    return FunctionType(fn.__code__, dict(module.__dict__, identity=identity, append_locked=frozen.append_line),
                        argdefs=fn.__defaults__)(spec, config, *args)


def conv_check(spec, config):
    """Real-bundle voxel reference, batch128 backward, memory and loss checks, no unused FMT."""
    e = frozen.engine; e.deterministic('cuda'); torch.manual_seed(96611)
    geometry, counts, labels = [], [], []
    for flow in spec['flows']:
        root = Path(spec['output'])/'physical'/flow['name']/'train'
        with np.load(root/'metadata.npz') as z:
            rows = np.r_[np.flatnonzero(z['labels'] == 0)[:16], np.flatnonzero(z['labels'] == 1)[:16]]
            counts.extend(z['counts'][rows]); labels.extend(z['labels'][rows])
        geometry.append(np.array(np.load(root/'geometry.npy', mmap_mode='r')[rows]))
    g = torch.as_tensor(np.concatenate(geometry), device='cuda'); c = torch.as_tensor(counts, device='cuda')
    assert torch.all(c == spec['neighbor_count']+1)
    off, ids, values = e.sparse_voxels(g, c, 16)
    x = e.dense_sparse_batch([dict(offsets=off, indices=ids, values=values)], [(0, i) for i in range(len(g))], 16, 'cuda')
    from FMT_Utils.Task4C_PaperBundles_3_1 import bundle_voxels
    from experiments.Task4C_ConvResolution_4_22 import reference_voxel
    torch.testing.assert_close(x, bundle_voxels(g, c, 16).half().float(), rtol=0, atol=0)
    ref = reference_voxel(g[0].cpu().numpy(), int(c[0]), 16)
    assert np.allclose(x[0].cpu().numpy(), ref, atol=.0005, rtol=.002)
    y = torch.as_tensor(labels, device='cuda'); model = e.Conv915().cuda()
    assert sum(p.numel() for p in model.parameters()) == 72192
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    model.eval()
    with torch.no_grad(): initial = float(e.F.cross_entropy(model(x), y))
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    for _ in range(30):
        model.train(); optimizer.zero_grad(set_to_none=True); loss = e.F.cross_entropy(model(x), y)
        assert torch.isfinite(loss); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad(): final = float(e.F.cross_entropy(model(x), y))
    assert final < initial
    model.train(); optimizer.zero_grad(set_to_none=True)
    e.F.cross_entropy(model(x.repeat(2, 1, 1, 1, 1)), y.repeat(2)).backward(); torch.cuda.synchronize()
    write(Path(spec['output'])/'preflight_cuda.json', dict(complete=True, identity=identity(config), engineering_only=True,
        parameters=72192, network=str(model), initial_loss=initial, final_loss=final, samples=len(y), batch128_backward=True,
        voxel_roundtrip_exact=True, independent_voxel_reference=True, seconds=time.perf_counter()-started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(), gpu=torch.cuda.get_device_name()))


def gpu_check(spec, config):
    assert json.loads((Path(spec['output'])/'data_audit.json').read_text())['identity'] == identity(config)
    if spec['family'] == 'conv': conv_check(spec, config)
    else:
        model_call(spec, config, 'preflight', 'cuda')
        model_call(spec, config, 'pilot')


def encode_conv(spec, config, index):
    """Frozen sparse-voxel encoder and cache format, omitting unused p35 descriptors."""
    e = frozen.engine; e.deterministic('cuda'); root = Path(spec['output']); name = spec['flows'][index]['name']
    assert json.loads((root/'preflight_cuda.json').read_text())['identity'] == identity(config)
    report = e.physical_reports(spec)[name]; records = {}; started = time.perf_counter()
    for split in frozen.SPLITS:
        src = root/'physical'/name/split
        for filename, digest in report['splits'][split]['files'].items(): assert sha(src/filename) == digest
        g = np.load(src/'geometry.npy', mmap_mode='r')
        with np.load(src/'metadata.npz') as z: counts = z['counts'].astype(np.int64)
        dest = root/'encoded'/name/split; dest.mkdir(parents=True, exist_ok=False)
        offsets, indices, values = [np.array([0], np.int64)], [], []
        for first in range(0, len(g), 32):
            sl = slice(first, first+32)
            off, ids, val = e.sparse_voxels(torch.as_tensor(np.array(g[sl]), device='cuda'), torch.as_tensor(counts[sl], device='cuda'), 16)
            offsets.append(off[1:]+offsets[-1][-1]); indices.append(ids); values.append(val)
        target = dest/'r16'; target.mkdir()
        for key, arrays in [('offsets', offsets), ('indices', indices), ('values', values)]: np.save(target/f'{key}.npy', np.concatenate(arrays))
        records[split] = dict(samples=len(g), files={str(f.relative_to(dest)): sha(f) for f in dest.rglob('*.npy')})
        write(dest/'manifest.json', records[split]); print('voxel16', name, split, len(g), flush=True)
    write(root/'encoding'/f'{index}.json', dict(complete=True, identity=identity(config), splits=records, seconds=time.perf_counter()-started))


def train(spec, config):
    for filename in ('data_audit.json', 'preflight_cuda.json'):
        report = json.loads((Path(spec['output'])/filename).read_text()); assert report['complete'] and report['identity'] == identity(config)
    if spec['family'] != 'conv':
        pilot = json.loads((Path(spec['output'])/'pilot.json').read_text()); assert pilot['complete'] and pilot['identity'] == identity(config)
    else:
        for index in range(2):
            encoded = json.loads((Path(spec['output'])/'encoding'/f'{index}.json').read_text()); assert encoded['complete'] and encoded['identity'] == identity(config)
    model_call(spec, config, 'train', 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['export', 'prepare', 'gpu-check', 'encode', 'train', 'summarize', 'runtime'])
    p.add_argument('--index', type=int, default=0); p.add_argument('--runtime-phase'); p.add_argument('--state'); p.add_argument('--exit-code', type=int)
    a = p.parse_args(); phase = a.runtime_phase if a.phase == 'runtime' else a.phase
    k = (6, 16)[a.index//(2 if phase in ('export', 'encode') else 3)] if phase != 'prepare' else (6,16)[a.index]
    config = CONFIGS[k]
    if a.phase == 'export': export(config, a.index % 2); return
    if a.phase == 'prepare':
        for family in FAMILIES: invoke('reuse', load_spec(config, family), config)
        return
    family = 'conv' if phase in ('encode', 'export', 'prepare') else FAMILIES[a.index % 3]
    spec = load_spec(config, family)
    if a.phase == 'gpu-check': gpu_check(spec, config)
    elif a.phase == 'encode': encode_conv(spec, config, a.index % 2)
    elif a.phase == 'train': train(spec, config)
    elif a.phase == 'summarize': invoke('summarize', spec, config)
    else: invoke('runtime', spec, config, a.runtime_phase, a.state, a.exit_code)


if __name__ == '__main__': main()
