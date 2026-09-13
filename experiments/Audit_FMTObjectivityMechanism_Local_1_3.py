"""Independent confusion-count audit of downloaded prediction evidence."""
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import numpy as np


def ratio(a,b):
    return float(a/b) if b else 0.


def main():
    root=Path('outputs/Verify_FMTObjectivityMechanism_1.3')
    records=list(csv.DictReader((root/'per_run_metrics.csv').open()))
    assert len(records)==1260
    by_run=collections.defaultdict(list)
    for r in records:by_run[(r['task'],r['dataset'],r['seed'])].append(r)
    devices=collections.Counter();events=[];parameters=set()
    with tarfile.open(root/'prediction_evidence.tar.gz') as archive:
        names=archive.getnames()
        assert not any(Path(n).suffix in {'.pt','.pth','.ckpt'} for n in names)
        for (task,dataset,seed),rows in by_run.items():
            prefix=f'shards/{task}/{dataset}/seed{seed}'
            data=archive.extractfile(prefix+'/predictions.npz').read()
            frozen=json.loads(archive.extractfile(prefix+'/frozen.json').read())
            assert len(rows)==(28 if task=='Task1' else 14)
            with np.load(io.BytesIO(data),allow_pickle=False) as z:
                truth=z['labels'].astype(bool)
                for r in rows:
                    pred=z[r['model']].astype(bool)
                    assert pred.shape==truth.shape
                    tp=int((pred&truth).sum());fp=int((pred&~truth).sum())
                    fn=int((~pred&truth).sum());tn=int((~pred&~truth).sum())
                    checks={'f1':ratio(2*tp,2*tp+fp+fn),'iou':ratio(tp,tp+fp+fn),
                            'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),
                            'positive_fraction':float(truth.mean()),
                            'predicted_positive_fraction':float(pred.mean())}
                    for key,value in checks.items():np.testing.assert_allclose(float(r[key]),value,atol=1e-12)
                    r.update(tp=tp,fp=fp,fn=fn,tn=tn)
            if task=='Task2':
                assert all(v['loss']['completed_optimizer_steps']==7000 for v in frozen.values())
                counts={v['loss']['parameter_count'] for v in frozen.values()};assert len(counts)==1
                parameters.update(counts)
        for name in names:
            if name.startswith('events/') and name.endswith('.json'):
                e=json.loads(archive.extractfile(name).read());events.append(e)
                if e['phase']=='Task2' and e['state']=='COMPLETED':devices[e['device']]+=1
    assert len(by_run)==60 and len(parameters)==1
    for summary in csv.DictReader((root/'summary.csv').open()):
        group=[float(r['f1']) for r in records if r['task']==summary['task'] and r['model']==summary['model']]
        assert len(group)==30
        np.testing.assert_allclose(np.mean(group),float(summary['macro_f1']),atol=1e-12)
    with (root/'independent_confusion_counts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    pairs=[]
    comparisons=[('old_full','old_no_center'),('old_full','old_recomputed_full'),
                 ('old_recomputed_no_center','neighbor_undifferenced_kin'),
                 ('old_no_center','gram6_kin'),('gram6_kin','gram6_unscaled_kin'),
                 ('gram6_kin','gram_full_kin'),('gram6_kin','gram_time_kin'),('gram6_kin','gram_delta6_kin')]
    for task in ['Task1','Task2']:
        lookup={(r['dataset'],r['seed'],r['model']):float(r['f1']) for r in records if r['task']==task}
        for a,b in comparisons:
            aa=a+('_pca8' if task=='Task1' else '');bb=b+('_pca8' if task=='Task1' else '')
            seeds=sorted({r['seed'] for r in records if r['task']==task})
            datasets=sorted({r['dataset'] for r in records if r['task']==task})
            deltas=[np.mean([lookup[d,s,bb]-lookup[d,s,aa] for d in datasets]) for s in seeds]
            pairs.append(dict(task=task,reference=aa,comparison=bb,mean_delta=float(np.mean(deltas)),
                              seed_macro_delta_sd=float(np.std(deltas,ddof=1))))
    result=dict(status='PASS',shards=len(by_run),metric_rows=len(records),
                vae_parameter_counts=list(parameters),gpu_shards_by_device=dict(devices),
                archive_sha256=hashlib.sha256((root/'prediction_evidence.tar.gz').read_bytes()).hexdigest(),
                audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                conclusion_scope='Confusion-count classification metrics, macro aggregation, paired seeds, capacity, step counts; not an independent silhouette recomputation.',
                paired_comparisons=pairs)
    (root/'local_independent_audit.json').write_text(json.dumps(result,indent=2))
    (root/'execution_events.json').write_text(json.dumps(events,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
