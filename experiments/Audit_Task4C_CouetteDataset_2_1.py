"""Independent all-raw-point label audit and reproducible curve/neighborhood probes."""
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from experiments import Build_Task4C_CouetteDataset_2_1 as build
from FMT_Utils.Task4C_HairpinBinary_2_1 import sample_gt
from FMT_Utils.Task4C_BallQuery_2_2 import select_fps


def main():
    build.initialize();s=build.SCENE;root=build.OUT
    with np.load(root/'physical/couette/metadata.npz') as z:meta=dict(z)
    seeds=np.load(root/'physical/couette/seeds.npy');curves=np.load(root/'physical/couette/curves.npy',mmap_mode='r')
    interpolators=[RegularGridInterpolator(tuple(s['axes'][::-1]),s[name],bounds_error=True) for name in ('velocity','omega')]
    expected=[];raw_total=0;checked=0
    for path in sorted((root/'chunks/final').glob('*.npz')):
        with np.load(path) as z:
            raw=z['raw_points'];offsets=z['raw_offsets'];rows=z['raw_rows'];raw_total+=len(raw);checked+=len(rows)
            owner,_=sample_gt(s['gt'],raw,s['locator']);np.testing.assert_array_equal(owner,z['raw_owners'])
            v,w=[fn(raw[:,::-1]) for fn in interpolators]
            n=(v*v).sum(1)*(w*w).sum(1);dot=(v*w).sum(1)
            head=(w[:,1]>0)&(n>0)&(2*dot*dot<n);np.testing.assert_array_equal(head,z['raw_head'])
            hc=np.full(len(z['short_count']),-1);lc=hc.copy()
            for row,a,b in zip(rows,offsets[:-1],offsets[1:]):
                hc[row]=np.sum((owner[a:b]>=0)&head[a:b]);lc[row]=np.sum((owner[a:b]>=0)&~head[a:b])
            np.testing.assert_array_equal(hc,z['head_count']);np.testing.assert_array_equal(lc,z['leg_count'])
            valid=z['valid'].all(1);labels=(z['short_count']>=17)&(hc>0)&(lc>0);expected.append(labels[valid])
    np.testing.assert_array_equal(np.concatenate(expected),meta['label'])
    # Reintegrate a fixed class-stratified probe; neither class is excluded from geometry checks.
    rng=np.random.default_rng(96611);probes=np.unique(np.r_[rng.choice(len(seeds),32,False),rng.choice(np.flatnonzero(meta['label']),32,False)])
    traced=build.rawcurl.trace_curves(s,seeds[probes])
    assert traced['valid'].all();np.testing.assert_array_equal(traced['curves'],curves[probes]);np.testing.assert_array_equal(traced['arcs'],meta['half_arc_lengths'][probes])
    # Full seed audit against independently gathered eight-corner candidate cells.
    axes=s['axes'];indices=[np.searchsorted(a,seeds[:,d],side='right')-1 for d,a in enumerate(axes)]
    ix,iy,iz=indices;lam=[];oyf=[]
    for z in (0,1):
        for y in (0,1):
            for x in (0,1):lam.append(s['lambda2'][iz+z,iy+y,ix+x]);oyf.append(s['oyf'][iz+z,iy+y,ix+x])
    assert (np.asarray(lam)<-.00125).all() and (np.asarray(oyf).astype(np.float64).mean(0)>0).all()
    radius=float(meta['poisson_radius']);dist=cKDTree(seeds).query(seeds,k=2,workers=8)[0][:,1]
    assert dist.min()>=radius*(1-1e-10)
    h=json.loads((root/'data_audit.json').read_text())['h'];nb=root/'neighbors/couette';initial=np.load(nb/'initial_count.npy');replays=0
    for k in (6,16):
        order=np.load(nb/f'order{k}.npy');radii=np.load(nb/f'effective_radius{k}.npy')
        for i in probes[:16]:
            d=np.linalg.norm(seeds-seeds[i],axis=1);d[i]=np.inf
            assert (d<=np.nextafter(3*h,np.inf)).sum()==initial[i]
            expected_radius=max(3*h,np.partition(d,k-1)[k-1]);np.testing.assert_allclose(expected_radius,radii[i],atol=1e-14,rtol=1e-12)
            ids=np.flatnonzero(d<=radii[i]+8*np.finfo(float).eps*max(1.,radii[i]))
            chosen=select_fps(seeds,seeds[i],ids,k) if len(ids)>k else ids
            np.testing.assert_array_equal(chosen,order[i]);replays+=1
    preserved=json.loads((root/'preserved_sources.json').read_text())
    for file,digest in preserved['files'].items():assert build.sha(build.Path(preserved['gallery_root'])/file)==digest
    report=dict(complete=True,version='Verify_Task4C_CouetteAudit_2.1',source_audit_sha256=build.sha(root/'data_audit.json'),
        samples=len(seeds),positive=int(meta['label'].sum()),raw_points_gt_requeried=raw_total,long_curves_checked=checked,
        independent_scipy_head_classification=True,all_labels_recomputed=True,all_seeds_strict_candidate=True,
        reintegrated_seeds=probes.tolist(),neighbor_full_scan_replays=replays,minimum_seed_distance=float(dist.min()),
        poisson_radius=radius,old_gallery_hashes_preserved=len(preserved['files']))
    build.write(root/'independent_audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
