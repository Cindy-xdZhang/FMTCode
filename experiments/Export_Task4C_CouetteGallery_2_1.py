"""Export actual new Couette data to the original train/test gallery schema."""
import json
from pathlib import Path
import numpy as np
from experiments.Build_Task4C_CouetteDataset_2_1 import OUT, ROOT, write, sha
from FMT_Utils.Task4C_Bundles_1_1 import resample_line


def describe(values):
    a=np.asarray(values)
    return dict(n=len(a),mean=float(a.mean()) if len(a) else 0.,
        quantiles=np.quantile(a,[0,.05,.5,.95,1]).tolist() if len(a) else [0.]*5)


def main():
    audit=json.loads((OUT/'data_audit.json').read_text());assert audit['complete']
    data=OUT/'physical/couette'
    for file,digest in audit['files'].items():assert sha(data/file)==digest
    seeds=np.load(data/'seeds.npy',mmap_mode='r');curves=np.load(data/'curves.npy',mmap_mode='r')
    with np.load(data/'metadata.npz') as z:meta=dict(z)
    neighbors=OUT/'neighbors/couette';initial=np.load(neighbors/'initial_count.npy')
    target=OUT/'gallery';target.mkdir(exist_ok=True)
    entry=dict(h=audit['h'],half_lengths=audit['half_lengths'],ds=audit['ds'],dataset_version=audit['version'],
        dataset_audit_sha256=sha(OUT/'data_audit.json'),splits={},preview_only=False,
        full_pool_samples=audit['samples'],full_pool_positive=audit['positive'],load_indices_audit_sha256=sha(OUT/'load_indices/audit.json'))
    instances=dict(instances={},samples={});chosen={}
    for ri,role in enumerate(('train','test')):
        available=np.load(OUT/f'load_indices/fold0/{role}_seed_indices.npy');rng=np.random.default_rng([96611,2,ri]);priority=rng.permutation(available)
        # Include each available instance/class in the display; never change the dataset or split.
        selected=[]
        for iid in np.unique(meta['instance'][available]):
            for label in (0,1):selected.extend(priority[(meta['instance'][priority]==iid)&(meta['label'][priority]==label)][:16].tolist())
        selected=np.asarray(selected,np.int64);remaining=priority[~np.isin(priority,selected)]
        selected=np.r_[selected,remaining[:max(0,256-len(selected))]];rng.shuffle(selected);chosen[role]=selected
        entry['splits'][role]=dict(population=len(available),positive=int(meta['label'][available].sum()),
            records=[dict(row=int(i),label=int(meta['label'][i]),fold=int(meta['fold'][i]),instance=int(meta['instance'][i]),
                assignment_kind=int(meta['assignment_kind'][i]),short_gt_count=int(meta['short_curve_gt_count'][i]),
                seed=seeds[i].tolist(),initial_count=int(initial[i])) for i in selected],neighbors={},population_stats={})
        for i in selected:
            ids,counts=np.unique(meta['short_curve_gt_owner'][i][meta['short_curve_gt_owner'][i]>=0],return_counts=True)
            instances['samples'][f'{role}:{i}']=dict(instance=int(meta['instance'][i]),label=int(meta['label'][i]),
                seed_gt_instance=int(meta['gt_owner'][i]),short_gt_count=int(meta['short_curve_gt_count'][i]),
                short_gt_instances={str(k):int(v) for k,v in zip(ids,counts)},
                assigned_instance_short_points=int(np.sum(meta['short_curve_gt_owner'][i]==meta['instance'][i])),
                longest_head_count=int(meta['longest_head_count'][i]),longest_leg_count=int(meta['longest_leg_count'][i]),
                longest_evaluated=bool(meta['longest_evaluated'][i]),longest_raw_point_count=int(meta['longest_raw_point_count'][i]))
    for k in (6,16):
        order=np.load(neighbors/f'order{k}.npy',mmap_mode='r');radius=np.load(neighbors/f'effective_radius{k}.npy');expanded=np.load(neighbors/f'expanded{k}.npy')
        for role,selected in chosen.items():
            selected_order=order[selected];dest=entry['splits'][role]
            for cls in ('all','0','1'):
                mask=np.zeros(len(seeds),bool);mask[np.load(OUT/f'load_indices/fold0/{role}_seed_indices.npy')]=True
                if cls!='all':mask&=meta['label']==int(cls)
                rows=np.flatnonzero(mask);n=len(rows)
                dest['population_stats'].setdefault(str(k),{})[cls]=dict(samples=n,initial_count=describe(initial[rows]),radius_h=describe(radius[rows]/audit['h']),
                    expanded_fraction=float(expanded[rows].mean()) if n else 0.,
                    opposite_split_fraction=float(((meta['fold'][order[rows]]==0)!=(meta['fold'][rows,None]==0)).mean()) if n else 0.,
                    different_instance_fraction=float((meta['instance'][order[rows]]!=meta['instance'][rows,None]).mean()) if n else 0.,
                    opposite_label_fraction=float((meta['label'][order[rows]]!=meta['label'][rows,None]).mean()) if n else 0.)
            ids=np.column_stack([selected,selected_order]);physical=np.asarray(curves[ids]).transpose(0,2,1,3,4)
            x=physical.astype(np.float64);center=x.mean((2,3),keepdims=True);r=np.linalg.norm(x-center,axis=-1).max((2,3))
            normalized=((x-center)/r[:,:,None,None,None]).astype(np.float32)
            wide=np.asarray([resample_line(line,48) for line in normalized.reshape(-1,32,3)]).reshape(len(selected),3,k+1,48,3)
            center48=wide.mean((2,3),keepdims=True);r48=np.linalg.norm(wide-center48,axis=-1).max((2,3))
            normalized48=(wide-center48)/r48[:,:,None,None,None]
            physical_seeds=np.broadcast_to(seeds[ids][:,None],(len(selected),3,k+1,3))
            seeds32=(physical_seeds-center[:,:,0])/r[:,:,None,None]
            seeds48=(seeds32-center48[:,:,0])/r48[:,:,None,None]
            files={}
            for name,array in [('physical32',physical),('c156_48',normalized48),('c156_seeds',seeds48)]:
                a=array.astype('<f4');assert np.isfinite(a).all();path=target/f'couette_{role}_k{k}_{name}.bin';a.tofile(path)
                files[name]=dict(path='couette/'+path.name,shape=list(a.shape),dtype='float32-little-endian',sha256=sha(path))
            dest['neighbors'][str(k)]=dict(files=files,ids=selected_order.tolist(),labels=meta['label'][selected_order].tolist(),
                folds=meta['fold'][selected_order].tolist(),instances=meta['instance'][selected_order].tolist(),seeds=seeds[selected_order].tolist(),
                effective_radius=radius[selected].tolist(),expanded=expanded[selected].tolist())
    for iid in range(6):
        source=ROOT/f'outputs/Verify_Task4C_GTHeadLeg_2.3/viewer/instances/couette_{iid}.json'
        mesh=json.loads(source.read_text(encoding='utf8'));path=target/f'instance_{iid}.json'
        write(path,dict(flow='couette',instance=iid,**mesh['original']))
        instances['instances'][str(iid)]=dict(path='couette/'+path.name,sha256=sha(path))
    write(target/'flow.json',entry);write(target/'instances.json',instances)
    write(target/'complete.json',dict(complete=True,source_data_audit_sha256=sha(OUT/'data_audit.json'),
        files={p.name:sha(p) for p in target.iterdir() if p.is_file() and p.name!='complete.json'},
        selection='display-only 16 per available instance/class, then fill to 256 from fixed random priority; no dataset resampling',
        samples={r:len(ids) for r,ids in chosen.items()},train_test_instances_disjoint=True))
    print(json.dumps(dict(complete=True,gallery=str(target))),flush=True)


if __name__=='__main__':main()
