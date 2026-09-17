import json,hashlib,os
from pathlib import Path
import numpy as np, torch
from experiments import Task4C_P35GTHead_1_1 as run
from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d
from FMT_Utils.FMT_P35_NormFrequency_3_1 import direction_spectrum
spec=run.load_spec(run.CONFIG)
run.frozen.engine.deterministic('cuda')
parts=[]
for flow in spec['flows']:
    folder=Path(spec['source_output'])/'physical'/flow['name']/'train'
    with np.load(folder/'metadata.npz') as z:
        ids=np.r_[np.flatnonzero(z['labels']==0)[:8],np.flatnonzero(z['labels']==1)[:8]]
        counts=z['counts'][ids].copy()
    g=np.array(np.load(folder/'geometry.npy',mmap_mode='r')[ids]);s=np.array(np.load(folder/'seeds.npy',mmap_mode='r')[ids])
    parts.append((g,s,counts))
g,s,c=[torch.tensor(np.concatenate([p[i] for p in parts])) for i in range(3)]
mask=torch.arange(27)[None]<c[:,None]
def nearest(s,c):
    d=torch.cdist(s,s);d.masked_fill_(~(torch.arange(27,device=s.device)[None]<c[:,None])[:,None,:],torch.inf)
    d.diagonal(dim1=1,dim2=2).fill_(torch.inf)
    return d.argsort(-1,stable=True)[...,:6]
def fixed(g,c,nn):
    center=torch.arange(27,device=g.device)[None,:,None].expand(len(g),-1,-1)
    ids=torch.cat((center,nn),-1);bi,li=(torch.arange(27,device=g.device)[None]<c[:,None]).nonzero(as_tuple=True)
    p=g[bi[:,None],ids[bi,li]]
    base=pathline_dft_features_3d(p,num_freq=6,neighbor_scale=100.,neighbor_weight=.5,return_numpy=False)
    vals=torch.cat((base[:,:23],direction_spectrum(p[:,0],6),run.frozen.method.pool_neighbors(base[:,23:],run.POOL)),-1)
    return vals
cpu_ids=nearest(s,c);gpu_ids=nearest(s.cuda(),c.cuda()).cpu()
rawcpu=run.encode_batch(g,s,c).cpu().numpy();rawgpu=run.encode_batch(g.cuda(),s.cuda(),c.cuda()).cpu().numpy()
batchgpu=torch.cat([run.encode_batch(g[i:i+8].cuda(),s[i:i+8].cuda(),c[i:i+8].cuda()) for i in range(0,32,8)]).cpu().numpy()
a=fixed(g,c,gpu_ids).numpy();b=fixed(g.cuda(),c.cuda(),gpu_ids.cuda()).cpu().numpy()
sets_cpu=cpu_ids.sort(-1).values;sets_gpu=gpu_ids.sort(-1).values
changed=((sets_cpu!=sets_gpu).any(-1)&mask)
d64=torch.cdist(s.double(),s.double());d64.diagonal(dim1=1,dim2=2).fill_(torch.inf)
gaps=[]
for bi,li in changed.nonzero().tolist():
    gaps.append(float(abs(d64[bi,li,cpu_ids[bi,li]].max()-d64[bi,li,gpu_ids[bi,li]].max())))
report=dict(job=os.environ.get('SLURM_JOB_ID'),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),valid_anchors=int(mask.sum()),different_order=int(((cpu_ids!=gpu_ids).any(-1)&mask).sum()),different_neighbor_sets=int(changed.sum()),max_selected_radius_difference_float64=max(gaps,default=0),cpu_gpu_features_max=float(abs(rawcpu-rawgpu).max()),gpu_batch_features_max=float(abs(rawgpu-batchgpu).max()),fixed_same_neighbor_ids_max=float(abs(a-b).max()),fixed_same_neighbor_ids_allclose=bool(np.allclose(a,b,atol=2e-4,rtol=2e-4)),gpu_batch_allclose=bool(np.allclose(rawgpu,batchgpu,atol=2e-4,rtol=2e-4)))
Path(spec['output'],'startup_diagnosis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
