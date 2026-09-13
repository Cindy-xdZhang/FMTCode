"""Verify scalar-distance identity and trace frozen Fourier inputs directly.

No IVD labels, no trained checkpoint, no edits to the frozen encoder.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from FMT_Utils import DFT_FMT_3D as frozen
from FMT_Utils.FMTAllV2_3D import fmt_all_v2


def pair_distances(x):
    i,j=np.triu_indices(7,1)
    return np.linalg.norm(x[:,i]-x[:,j],axis=-1)


def scalar_fourier(distances):
    f=np.fft.rfft(distances,axis=2)[:,:,:6]
    return np.concatenate([f.real.reshape(len(f),-1),f.imag[:,:,1:].reshape(len(f),-1)],axis=1)


def observer(x,kind):
    t=np.arange(x.shape[2]);q=np.zeros((len(t),3,3))
    if kind=='exact_quarter_turn':
        c=np.array([1,0,-1,0])[t%4];s=np.array([0,1,0,-1])[t%4]
        shift=np.stack([t,2*t,-t],axis=1)
    else:
        a=1.3*t/max(len(t)-1,1);c=np.cos(a);s=np.sin(a)
        shift=np.stack([a,a*a,np.sin(a)],axis=1)
    q[:,0,0]=c;q[:,0,1]=-s;q[:,1,0]=s;q[:,1,1]=c;q[:,2,2]=1
    # The SAME Q(t), c(t) act on EVERY material particle; no re-seeding.
    return np.einsum('tij,nktj->nkti',q,x)+shift[None,None]


def trace(x):
    original=frozen.dft_rotation_invariants_3d
    calls=[]
    def capture(seq,*args,**kwargs):
        calls.append(seq.detach().cpu().numpy().copy())
        return original(seq,*args,**kwargs)
    frozen.dft_rotation_invariants_3d=capture
    try:
        f=frozen.pathline_dft_features_3d(torch.from_numpy(x),neighbor_scale=1.,neighbor_weight=1.)
    finally:
        frozen.dft_rotation_invariants_3d=original
    d=x[:,1:]-x[:,:1]
    expected=np.diff(d,axis=2).reshape(-1,x.shape[2]-1,3)
    np.testing.assert_array_equal(calls[1],expected)
    return f,calls


def difference(a,b):
    return dict(max_absolute=float(np.max(np.abs(a-b))),
                relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30)),
                bitwise_equal=bool(np.array_equal(a,b)))


def fixture():
    rng=np.random.default_rng(12091)
    x=rng.integers(-30,31,size=(64,7,32,3)).astype(np.float64)
    # Integer fixture plus exact signed-permutation rotations eliminates
    # floating-point distance error, testing the identity premise literally.
    return x


def check(name,x,kind):
    y=observer(x,kind)
    d,e=pair_distances(x),pair_distances(y)
    a,b=scalar_fourier(d),scalar_fourier(e)
    old,inputs=trace(x);changed,changed_inputs=trace(y)
    g,h=fmt_all_v2(x),fmt_all_v2(y)
    scaler=StandardScaler().fit(a)
    z,w=scaler.transform(a),scaler.transform(b)
    pca=PCA(n_components=8,svd_solver='full').fit(z)
    z,w=pca.transform(z),pca.transform(w)
    model=KMeans(n_clusters=2,n_init=20,random_state=12091).fit(z)
    pred,other=model.predict(z),model.predict(w)
    torch.manual_seed(12091)
    net=torch.nn.Sequential(torch.nn.Linear(a.shape[1],32),torch.nn.Tanh(),torch.nn.Linear(32,8)).double().eval()
    with torch.no_grad():
        out=net(torch.from_numpy(a)).numpy();out_other=net(torch.from_numpy(b)).numpy()
        replay=net(torch.from_numpy(a.copy())).numpy()
    assert np.array_equal(out,replay)
    result=dict(dataset=name,observer=kind,n=len(x),points_per_line=x.shape[2],
                distance=difference(d,e),distance_time_difference=difference(np.diff(d,axis=2),np.diff(e,axis=2)),
                distance_fourier=difference(a,b),fixed_demo_network=difference(out,out_other),
                identical_array_network_replay_max=float(np.max(np.abs(out-replay))),
                fixed_distance_pipeline_changed_clusters=int((pred!=other).sum()),
                actual_old_neighbor_fourier_input_shape=list(inputs[1].shape),
                actual_old_neighbor_fourier_input=difference(inputs[1],changed_inputs[1]),
                old_neighbor_output=difference(old[:,23:],changed[:,23:]),
                objective_gram_output=difference(g,h),
                traced_input_equals_three_coordinate_differences=True,
                fixed_demo_network_is_not_trained=True,
                labels_used=False)
    if kind=='exact_quarter_turn':
        for key in ['distance','distance_time_difference','distance_fourier','fixed_demo_network']:
            assert result[key]['bitwise_equal']
        assert result['fixed_distance_pipeline_changed_clusters']==0
        assert result['actual_old_neighbor_fourier_input']['max_absolute']>0
        assert result['old_neighbor_output']['max_absolute']>0
    else:
        np.testing.assert_allclose(d,e,rtol=1e-10,atol=1e-11)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='C:/Users/xingdi/sources/FLowlineClusteringVisualAnalysis/outputs/Other_FeatureSilhouette_1.1')
    p.add_argument('--output',default='outputs/Verify_DistanceInputIdentity_1.1')
    args=p.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        rows=[check('integer_identity_fixture',fixture(),'exact_quarter_turn')]
        for file in sorted(Path(args.root).glob('*/geometry/geometry_bundle.npz')):
            with np.load(file,allow_pickle=False) as z:
                x=np.asarray(z['paths'],dtype=np.float64)
                meta=json.loads(str(z['metadata_json']))
            result=check(file.parent.parent.name,x,'smooth_rotation_translation')
            result['source_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
            result['source_time']=meta.get('start_time')
            rows.append(result)
            print(json.dumps(result),flush=True)
        assert len(rows)==9
    result=dict(experiment='Verify_DistanceInputIdentity_1.1',status='PASS',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                frozen_encoder_sha256=hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest(),
                numpy=np.__version__,torch=torch.__version__,rows=rows)
    (root/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PASS: exact identity fixture and all eight real-data populations; no labels used')


if __name__=='__main__':main()
