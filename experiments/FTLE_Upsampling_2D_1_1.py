"""Reproducible migration of FTLE upsampling to current five-line FMT encoders."""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.ndimage import binary_erosion, uniform_filter
from torch import nn

from FMT_Utils.FTLE_Baselines_2D import ESPCN_SR, UNet_SR
from FMT_Utils.FTLE_Data_2D import (Velocity, evaluation_mask, file_sha256, ftle, integrate,
                                     interpolation, metadata, time_splits)
from FMT_Utils.FTLE_Encoders_2D import encode

GEOMETRY = ('raw', 'fmt', 'fmt_objective_ntod_v2', 'fmt_v5')


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def git_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def provenance(config):
    return {'commit': git_commit(), 'config_sha256': file_sha256(config), 'timestamp': stamp(),
            'hostname': socket.gethostname(), 'job_id': os.environ.get('SLURM_JOB_ID'),
            'array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
            'torch': torch.__version__, 'numpy': np.__version__,
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'source_sha256': {p:file_sha256(p) for p in (
                'experiments/FTLE_Upsampling_2D_1_1.py', 'experiments/Audit_FTLE_Upsampling_2D_1_1.py',
                'FMT_Utils/FTLE_Data_2D.py','FMT_Utils/FTLE_Encoders_2D.py','FMT_Utils/FTLE_Baselines_2D.py',
                'FMT_Utils/DCT_FMT_encoder.py','FMT_Utils/DCT_utils.py')}}


class Model(nn.Module):
    def __init__(self, spec, method, scale, width=0):
        super().__init__()
        self.method, self.scale = method, scale
        cfg = SimpleNamespace(model=SimpleNamespace(**spec['model']))
        self.padded = spec['model']['padded_geometry_channels']
        if method == 'espcn':
            self.net = ESPCN_SR(cfg, upscale=scale, in_ch=1)
        elif method == 'unet':
            self.net = UNet_SR(cfg, upscale=scale, in_ch=1)
        else:
            if width > self.padded or width < 1:
                raise ValueError('Unexpected geometry feature width')
            projection = spec['model']['geometry_projection']
            self.project = nn.Conv2d(self.padded, projection, 1)
            self.net = UNet_SR(cfg, upscale=scale, in_ch=1 + projection)
            self.net.head = nn.Sequential(nn.Conv2d(spec['model']['base'], scale*scale, 3, padding=1),
                                          nn.PixelShuffle(scale))

    def forward(self, low, geometry=None):
        value = low
        if self.method in GEOMETRY:
            if geometry is None or geometry.shape[1] > self.padded:
                raise ValueError('Missing or oversized geometry')
            padded = torch.nn.functional.pad(geometry, (0, 0, 0, 0, 0, self.padded-geometry.shape[1]))
            value = torch.cat((low, self.project(padded)), dim=1)
        pred = self.net(value)
        # Low sites are high indices 0,s,2s,..., including both endpoints.
        return pred[..., :(low.shape[-2]-1)*self.scale+1, :(low.shape[-1]-1)*self.scale+1]


def prepare(spec, config, index, device):
    flow = spec['flows'][index]
    root = Path(spec['output']) / 'data' / flow['name']
    if (root / 'manifest.json').exists():
        raise FileExistsError('Prepared data are immutable; use a new output/version')
    root.mkdir(parents=True, exist_ok=True)
    meta = metadata(spec['data_root'], flow['name'])
    splits = time_splits(meta, spec['split_counts'], spec['integration']['tau'])
    xmin, xmax, ymin, ymax = meta['bounds']
    offset = spec['integration']['offset']
    height, width = flow['high_shape']
    if any((n-1) % s for n in (height, width) for s in spec['scales']):
        raise ValueError('High grid intervals must be divisible by every scale')
    xs = np.linspace(xmin + 2*offset, xmax - 2*offset, width)
    ys = np.linspace(ymin + 2*offset, ymax - 2*offset, height)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    seeds = np.stack((xx, yy), -1).reshape(-1, 2)
    records = []
    for split, times in splits.items():
        for slice_index, t0 in enumerate(times):
            start = time.monotonic()
            velocity = Velocity(meta, t0, spec['integration']['tau'], device)
            paths, valid, integration = integrate(velocity, seeds, t0, **spec['integration'])
            truth = np.zeros(len(seeds), dtype=np.float32)
            truth[valid.cpu().numpy()] = ftle(paths[valid], spec['integration']['tau']).cpu().numpy()
            truth = truth.reshape(height, width)
            valid_high = valid.cpu().numpy().reshape(height, width)
            structured = paths.view(height, width, 5, spec['integration']['samples'], 2)
            for scale in spec['scales']:
                low = truth[::scale, ::scale].copy()
                low_valid = valid_high[::scale, ::scale].copy()
                paths_low = structured[::scale, ::scale].reshape(-1, 5, spec['integration']['samples'], 2)
                mask = low_valid.ravel()
                arrays = {'low': low, 'high': truth, 'valid_high': valid_high, 'valid_low': low_valid,
                          'mask': evaluation_mask(valid_high, low_valid, scale), 'xs': xs, 'ys': ys,
                          'times': np.linspace(t0, t0 + spec['integration']['tau'], spec['integration']['samples'])}
                for method in GEOMETRY:
                    feature_parts = [encode(part, method, spec['fourier_bins']).cpu().numpy()
                                     for part in paths_low[torch.from_numpy(mask).to(device)].split(2048)]
                    features = np.concatenate(feature_parts)
                    full = np.zeros((len(paths_low), features.shape[1]), np.float32)
                    full[mask] = features
                    arrays[method] = full.reshape(*low.shape, -1).transpose(2, 0, 1)
                # Store only the low-resolution five-line geometry as audit evidence.
                arrays['paths_low'] = paths_low.cpu().numpy()
                name = f'{split}_{slice_index:02d}_x{scale}.npz'
                path = root / name
                np.savez_compressed(path, **arrays)
                records.append({'file': name, 'sha256': file_sha256(path), 'split': split, 'scale': scale,
                                'slice_index': slice_index, 'valid_high': int(valid_high.sum()),
                                'evaluated_pixels': int(arrays['mask'].sum()), 'integration': integration})
            del velocity, paths, structured, paths_low
            print(json.dumps({'flow': flow['name'], 'split': split, 'slice': slice_index,
                              'valid_fraction': float(valid_high.mean()), 'seconds': time.monotonic()-start}), flush=True)
    dump(root / 'manifest.json', {'version': spec['version'], 'provenance': provenance(config),
                                 'source': meta, 'time_splits': splits, 'records': records})


def load_data(spec, flow, scale, split):
    root = Path(spec['output']) / 'data' / flow
    manifest = read(root / 'manifest.json')
    result = []
    for row in manifest['records']:
        if row['split'] == split and row['scale'] == scale:
            with np.load(root / row['file']) as ds:
                # Never load high-resolution geometry: none is stored in this cache.
                result.append({**{k: ds[k] for k in ('low','high','valid_high','valid_low','mask', *GEOMETRY)},
                               'file': row['file']})
    if len(result) != spec['split_counts'][split]:
        raise ValueError('Incomplete split')
    return result


def normalization(data, method):
    high = np.concatenate([d['high'][d['valid_high']] for d in data]).astype(np.float64)
    low = np.concatenate([d['low'][d['valid_low']] for d in data]).astype(np.float64)
    result = {'low_mean': float(low.mean()), 'low_std': max(float(low.std()), 1e-6),
              'target_mean': float(high.mean()), 'target_std': max(float(high.std()), 1e-6),
              'data_range': max(float(np.ptp(high)), 1e-6)}
    if method in GEOMETRY:
        count, total, square = 0, None, None
        for d in data:
            x = d[method][:, d['valid_low']].astype(np.float64)
            count += x.shape[1]
            total = x.sum(1) if total is None else total + x.sum(1)
            square = (x*x).sum(1) if square is None else square + (x*x).sum(1)
        mean = total / count
        std = np.sqrt(np.maximum(square/count - mean*mean, 0)).clip(1e-6)
        result.update(feature_mean=mean.tolist(), feature_std=std.tolist())
    return result


def normalized(data, method, stats):
    result = []
    for d in data:
        low = (d['low'] - stats['low_mean']) / stats['low_std']
        low[~d['valid_low']] = 0
        geo = None
        if method in GEOMETRY:
            geo = ((d[method] - np.asarray(stats['feature_mean'])[:,None,None]) /
                   np.asarray(stats['feature_std'])[:,None,None]).astype(np.float32)
            geo[:, ~d['valid_low']] = 0
        result.append({**d, 'input': low[None].astype(np.float32), 'geometry': geo,
                       'target': ((d['high'] - stats['target_mean'])/stats['target_std'])[None].astype(np.float32)})
    return result


def patch_indices(data, spec, scale):
    size = spec['training']['patch_high_intervals']//scale + 1
    stride = spec['training']['patch_stride_high']//scale
    out = []
    for index, d in enumerate(data):
        h, w = d['low'].shape
        rows = sorted(set(range(0, h-size+1, stride)) | {h-size})
        cols = sorted(set(range(0, w-size+1, stride)) | {w-size})
        for y, x in itertools.product(rows, cols):
            if min(y, x) < 0:
                continue
            high = d['valid_high'][y*scale:(y+size-1)*scale+1, x*scale:(x+size-1)*scale+1]
            if high.all() and d['valid_low'][y:y+size,x:x+size].all():
                out.append((index,y,x))
    if not out:
        raise ValueError('No wholly valid training patches')
    return out, size


def batch(data, indices, size, scale, device):
    low, geo, high = [], [], []
    for index, y, x in indices:
        d = data[index]
        low.append(d['input'][:,y:y+size,x:x+size])
        if d['geometry'] is not None:
            geo.append(d['geometry'][:,y:y+size,x:x+size])
        high.append(d['target'][:,y*scale:(y+size-1)*scale+1,x*scale:(x+size-1)*scale+1])
    return (torch.from_numpy(np.stack(low)).to(device),
            torch.from_numpy(np.stack(geo)).to(device) if geo else None,
            torch.from_numpy(np.stack(high)).to(device))


@torch.no_grad()
def predict(model, data, stats, device):
    model.eval()
    predictions = []
    for d in data:
        low = torch.from_numpy(d['input'][None]).to(device)
        geo = torch.from_numpy(d['geometry'][None]).to(device) if d['geometry'] is not None else None
        output = model(low, geo)[0,0].cpu().numpy()
        predictions.append(output * stats['target_std'] + stats['target_mean'])
    return predictions


def metrics(truth, prediction, mask, data_range):
    truth, prediction = truth.astype(np.float64), prediction.astype(np.float64)
    if not mask.any() or not np.isfinite(prediction).all():
        raise ValueError('No valid evaluation support or nonfinite prediction')
    delta = prediction[mask] - truth[mask]
    mse = float(np.mean(delta**2))
    mx, my = uniform_filter(truth, 7), uniform_filter(prediction, 7)
    vx = (uniform_filter(truth*truth,7)-mx*mx)*49/48
    vy = (uniform_filter(prediction*prediction,7)-my*my)*49/48
    cov = (uniform_filter(truth*prediction,7)-mx*my)*49/48
    c1, c2 = (.01*data_range)**2, (.03*data_range)**2
    ssim = ((2*mx*my+c1)*(2*cov+c2))/((mx*mx+my*my+c1)*(vx+vy+c2))
    smask = binary_erosion(mask, structure=np.ones((7,7),bool), border_value=0)
    return {'mse': mse, 'rmse': float(np.sqrt(mse)), 'mae': float(np.mean(abs(delta))),
            'psnr': float(10*np.log10(data_range**2/max(mse,1e-30))),
            'ssim': float(ssim[smask].mean()) if smask.any() else None,
            'pixels': int(mask.sum()), 'ssim_pixels': int(smask.sum())}


def validation_mse(data, predictions):
    return float(np.mean([np.mean((p[d['mask']].astype(np.float64)-d['high'][d['mask']])**2)
                          for d,p in zip(data,predictions)]))


def matrix(spec):
    return list(itertools.product([d['name'] for d in spec['flows']], spec['scales'], spec['seeds']))


def train(spec, config, index, device):
    flow, scale, seed = matrix(spec)[index]
    train_raw = load_data(spec, flow, scale, 'train')
    val_raw = load_data(spec, flow, scale, 'validation')
    for method in spec['methods']:
        root = Path(spec['output'])/'runs'/f'{flow}_x{scale}_s{seed}'/method
        if (root/'result.json').exists():
            raise FileExistsError('Results already exist; do not silently overwrite/retest')
        root.mkdir(parents=True, exist_ok=True)
        stats = normalization(train_raw, method)
        tr, va = normalized(train_raw, method, stats), normalized(val_raw, method, stats)
        indices, size = patch_indices(tr, spec, scale)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        width = len(stats.get('feature_mean', []))
        model = Model(spec, method, scale, width).to(device)
        cfg = spec['training']
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.5)
        rng = np.random.default_rng(seed)
        best, best_epoch, state = float('inf'), -1, None
        history = []
        start = time.monotonic()
        for epoch in range(cfg['epochs']):
            model.train()
            order = rng.permutation(len(indices))
            losses = []
            for begin in range(0, len(order), cfg['batch_size']):
                subset = [indices[i] for i in order[begin:begin+cfg['batch_size']]]
                low, geo, target = batch(tr, subset, size, scale, device)
                optimizer.zero_grad(set_to_none=True)
                loss = (model(low, geo)-target).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward(); optimizer.step()
                losses.append((float(loss.detach()), len(subset)))
            predictions = predict(model, va, stats, device)
            score = validation_mse(va, predictions)
            history.append({'epoch': epoch, 'train_normalized_mse': sum(a*b for a,b in losses)/sum(b for _,b in losses),
                            'validation_mse': score, 'lr': optimizer.param_groups[0]['lr']})
            if score < best:
                best, best_epoch = score, epoch
                state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            scheduler.step(score)
            dump(root/'history.json', history)
            if epoch % 10 == 0:
                print(json.dumps({'flow':flow,'scale':scale,'seed':seed,'method':method,'epoch':epoch,'validation_mse':score}), flush=True)
            if epoch-best_epoch >= cfg['patience']:
                break
        model.load_state_dict(state)
        # This is the first test-data read in a method run, after selection is final.
        te = normalized(load_data(spec, flow, scale, 'test'), method, stats)
        rows = []
        for split, data in [('validation', va), ('test', te)]:
            for d, pred in zip(data, predict(model, data, stats, device)):
                filename = f'{split}_{d["file"]}'
                np.savez_compressed(root/filename, prediction=pred, truth=d['high'], mask=d['mask'])
                rows.append({'split':split,'file':filename,'source_file':d['file'],
                             **metrics(d['high'],pred,d['mask'],stats['data_range'])})
        result = {'version':spec['version'],'flow':flow,'scale':scale,'seed':seed,'method':method,
                  'provenance':provenance(config),'parameters':sum(p.numel() for p in model.parameters()),
                  'feature_width':width,'patch_count':len(indices),'patch_size_low':size,'normalization':stats,
                  'best_epoch':best_epoch,'best_validation_mse':best,'epochs_completed':len(history),
                  'seconds':time.monotonic()-start,'metrics':rows,'checkpoint_files':[]}
        dump(root/'result.json', result)
        del model, state, optimizer, tr, va, te
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(json.dumps({'completed':str(root),'best_epoch':best_epoch}), flush=True)


def summarize(spec, config):
    rows = []
    for flow, scale, seed in matrix(spec):
        for method in spec['methods']:
            result = read(Path(spec['output'])/'runs'/f'{flow}_x{scale}_s{seed}'/method/'result.json')
            for row in result['metrics']:
                rows.append({**{k:result[k] for k in ('flow','scale','seed','method','parameters')}, **row})
    # Interpolation is deterministic; report once instead of pretending it has three seeds.
    for flow, scale in itertools.product([d['name'] for d in spec['flows']],spec['scales']):
        stats = normalization(load_data(spec,flow,scale,'train'),'unet')
        for split in ('validation','test'):
            for d in load_data(spec,flow,scale,split):
                for method,order in [('bilinear',1),('bicubic',3)]:
                    rows.append({'flow':flow,'scale':scale,'seed':-1,'method':method,'parameters':0,
                                 'split':split,'file':d['file'],'source_file':d['file'],
                                 **metrics(d['high'],interpolation(d['low'],scale,order),d['mask'],stats['data_range'])})
    path = Path(spec['output'])/'metrics.csv'
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    aggregate = []
    for flow,scale,method in itertools.product([d['name'] for d in spec['flows']],spec['scales'],spec['methods']+['bilinear','bicubic']):
        group = [r for r in rows if r['flow']==flow and r['scale']==scale and r['method']==method and r['split']=='test']
        item = {'flow':flow,'scale':scale,'method':method,'seeds':len(set(r['seed'] for r in group))}
        for metric in ('rmse','mae','psnr','ssim'):
            means = [np.mean([r[metric] for r in group if r['seed']==seed and r[metric] is not None])
                     for seed in sorted(set(r['seed'] for r in group))]
            item[metric+'_mean'] = float(np.mean(means)); item[metric+'_std'] = float(np.std(means))
        aggregate.append(item)
    dump(Path(spec['output'])/'summary.json',{'version':spec['version'],'provenance':provenance(config),
                                             'aggregation':'equal slice means, then seed mean and population standard deviation',
                                             'results':aggregate})


def event(spec, config, phase, state, code=None):
    record = {'version':spec['version'],'phase':phase,'state':state,'exit_code':code,**provenance(config)}
    root = Path(spec['output'])/'runtime'
    root.mkdir(parents=True,exist_ok=True)
    identifier = f'{os.environ.get("SLURM_JOB_ID","local")}_{os.environ.get("SLURM_ARRAY_TASK_ID","none")}_{phase}_{state}'
    dump(root/(identifier+'.json'),record)


def submit(spec, config):
    root = Path(spec['output']); (root/'slurm').mkdir(parents=True,exist_ok=True)
    jobs = []
    def dispatch(phase, count, dependency=None, gpu=False):
        cmd = ['sbatch','--parsable','--cpus-per-task=4','--mem=48G','--time=06:00:00',
               '--job-name=ftle11-'+phase,'--output='+str(root/'slurm'/(phase+'-%A_%a.log'))]
        if count > 1: cmd += ['--array',f'0-{count-1}%8']
        if gpu: cmd += ['--gres=gpu:1','--constraint=a100|v100']
        if dependency: cmd += ['--dependency=afterok:'+dependency]
        cmd += ['ibex_bash/ftle_upsampling_2d_1p1.sh',phase,config]
        submitted = stamp()
        job = subprocess.check_output(cmd,text=True).strip().split(';')[0]
        record = {'job_id':job,'version':spec['version'],'phase':phase,'submitted_utc':submitted,
                  'config':config,'config_sha256':file_sha256(config),'commit':git_commit(),
                  'expected_device':'A100 or V100' if gpu else 'CPU','array_count':count,'command':cmd}
        jobs.append(record); dump(root/'submission.json',jobs)
        # Append immediately; preserve all failed/cancelled IDs as well.
        with open('docs/ibex_run_registry.md','a',encoding='utf-8') as stream:
            stream.write('\n\nFTLE 2D 1.1 submission: `'+json.dumps(record)+'`\n')
        print(json.dumps(record),flush=True)
        return job
    check=dispatch('preflight',1)
    prep=dispatch('prepare',len(spec['flows']),check,True)
    data_audit=dispatch('audit-data',1,prep)
    fit=dispatch('train',len(matrix(spec)),data_audit,True)
    merged=dispatch('summarize',1,fit)
    dispatch('audit-results',1,merged)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['preflight','prepare','train','summarize','audit-data','audit-results','submit','runtime'])
    parser.add_argument('--config',default='config/Other_FTLEUpsampling2D_1.1.json')
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--runtime-phase'); parser.add_argument('--state'); parser.add_argument('--exit-code',type=int)
    args=parser.parse_args(); spec=read(args.config)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    if args.phase=='runtime': event(spec,args.config,args.runtime_phase,args.state,args.exit_code)
    elif args.phase=='prepare': prepare(spec,args.config,args.index,args.device)
    elif args.phase=='train': train(spec,args.config,args.index,args.device)
    elif args.phase=='summarize': summarize(spec,args.config)
    elif args.phase=='submit': submit(spec,args.config)
    else:
        from experiments.Audit_FTLE_Upsampling_2D_1_1 import audit_data,audit_results,preflight
        {'preflight':preflight,'audit-data':audit_data,'audit-results':audit_results}[args.phase](spec,args.config)


if __name__=='__main__':
    main()
