"""Independently audit saliency geometry, display selection and replay evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def audit(root):
    root=Path(root);package=root/'package'
    manifest=json.loads((package/'manifest.json').read_text())
    spec=manifest['config'];source=Path(spec['source_output'])
    source_run=source/'runs/p35'/f"seed{spec['seed']}"
    run=root/'runs/p35'/f"seed{spec['seed']}"
    old=json.loads((source_run/'result.json').read_text());new=json.loads((run/'result.json').read_text())
    assert new['selected_epoch']==old['selected_epoch']==56 and new['epochs']==56
    fields=('epoch','training_loss','validation_loss','validation_f1','validation_average_precision',
            'learning_rate','samples','permutation_sha256')
    for a,b in zip(new['history'],old['history']):
        assert all(a[k]==b[k] for k in fields)
    comparisons={}
    for split in ('train','validation','test'):
        with np.load(source_run/f'{split}_predictions.npz') as a,np.load(run/f'{split}_predictions.npz') as b:
            assert a.files==b.files
            assert all(np.array_equal(a[k],b[k]) for k in a.files),split
            y=b['labels']>0;p=b['probability']>=.5
            tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum())
            f1=2*tp/(2*tp+fp+fn)
            assert f1==new['metrics'][split]['combined']['f1']==old['metrics'][split]['combined']['f1']
            comparisons[split]=dict(rows=len(y),all_prediction_arrays_exact=True,f1=f1)
    with np.load(source_run/'test_predictions.npz') as z:pred={k:z[k] for k in z.files}
    totals={};patches=points=0;probability_error=0.
    for fi,flow in enumerate(('channel','tbl')):
        physical=source/'physical'/flow/'test'
        with np.load(physical/'metadata.npz') as z:m={k:z[k] for k in z.files}
        g=np.load(physical/'geometry.npy',mmap_mode='r')
        probability=pred['probability'][pred['flow_index']==fi]
        assert np.array_equal(pred['row_in_split'][pred['flow_index']==fi],np.arange(len(g)))
        eligible=np.flatnonzero(probability>=.5);x=m['center'][eligible].astype(np.float64)
        span=np.ptp(x,axis=0);span[span==0]=1.;x=(x-x.min(0))/span
        index=int(np.random.default_rng(spec['selection_seed']+fi).integers(len(x)))
        selected=[];distance=np.full(len(x),np.inf)
        for _ in range(spec['examples_per_flow']):
            selected.append(index);distance=np.minimum(distance,np.square(x-x[index]).sum(1))
            distance[selected]=-1;index=int(distance.argmax())
        records=[r for r in manifest['records'] if r['flow']==flow]
        assert np.array_equal(eligible[selected],[r['row'] for r in records])
        tp=fp=0
        for r in records:
            row=r['row'];n=int(m['counts'][row]);assert n==r['count']
            assert r['label']==int(m['labels'][row]) and r['probability']==float(probability[row])
            if r['label']:tp+=1
            else:fp+=1
            file=package/r['file'];assert sha(file)==r['sha256']
            with np.load(file) as z:
                assert np.array_equal(z['normalized_geometry'],g[row])
                expected=np.asarray(g[row],np.float64)*float(m['radius'][row])+m['centroid'][row]
                expected[n:]=0
                assert np.array_equal(z['geometry'],expected)
                ids=z['neighbors'][:n];assert np.array_equal(ids[:,0],np.arange(n))
                assert np.all((ids>=0)&(ids<n)) and all(len(np.unique(a))==7 for a in ids)
                probability_error=max(probability_error,abs(float(z['probability'])-r['probability']))
                assert probability_error<=1e-4
                expected_patches=np.asarray([(line,a,b) for line in range(n) for a,b in spec['patch_intervals']])
                assert np.array_equal(z['patch_indices'],expected_patches)
                displayed=np.zeros((27,32));count=np.zeros((27,32))
                for (line,a,b),change in zip(z['patch_indices'],z['patch_margin_delta']):
                    displayed[line,a+1:b]+=change;count[line,a+1:b]+=1
                assert np.array_equal(displayed/np.maximum(count,1),z['local_shape_delta'])
                for key in ('gradient','smooth_gradient','local_shape_delta'):
                    assert np.isfinite(z[key]).all() and np.count_nonzero(z[key][n:])==0
                points+=n*32;patches+=len(expected_patches)
        totals[flow]=dict(bundles=len(records),true_positive=tp,false_positive=fp)
    runtime=[json.loads(line) for line in (root/'runtime.jsonl').read_text().splitlines()]
    jobs={}
    for row in runtime:jobs.setdefault(row['job'],[]).append(row)
    assert len(jobs)==4 and all(len(v)==2 for v in jobs.values())
    assert sum(v[-1]['exit_code']==0 for v in jobs.values())==3
    assert not any(root.rglob('*.pt')) and not any(root.rglob('*.pth')) and not any(root.rglob('*.ckpt'))
    report=dict(complete=True,version=spec['version'],scientific_identity=manifest['identity'],
        auditor_sha256=sha(__file__),manifest_sha256=sha(package/'manifest.json'),
        replay_comparison=comparisons,all_56_epoch_values_exact=True,
        geometry_matches_original_arrays=True,physical_coordinates_exact=True,
        selection_recomputed_without_labels=True,point_averages_independently_recomputed=True,
        bundles=len(manifest['records']),points=points,patch_interventions=patches,flows=totals,
        max_geometry_forward_probability_error=probability_error,
        jobs={job:dict(states=[r['state'] for r in rows],exit_code=rows[-1]['exit_code'],
                      node=rows[0]['host'],gpu=rows[0].get('gpu')) for job,rows in jobs.items()},
        no_weights=True)
    (root/'independent_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    audit(parser.parse_args().root)
