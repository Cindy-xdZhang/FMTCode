"""Independent confusion-matrix and prediction audit of all completed shards."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, adjusted_rand_score, normalized_mutual_info_score


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config/Verify_FMTAllV2_1.1.json')
    args=parser.parse_args()
    spec=json.loads(Path(args.config).read_text());root=Path(spec['output_root'])
    checked=[]
    for task in spec['tasks']:
        for dataset in spec['datasets']:
            reference_hash=None
            for seed in spec[task.lower()]['seeds']:
                folder=root/'shards'/task/dataset/f'seed{seed}'
                done=json.loads((folder/'complete.json').read_text())
                assert done['status']=='COMPLETE'
                assert done['prediction_sha256']==sha(folder/'predictions.npz')
                audit=json.loads((folder/'input_audit.json').read_text())
                confirmation=json.loads((folder/'confirmation_input_audit.json').read_text())
                groups=[{x['path'] for x in rows} for rows in (audit['training'],audit['validation'],confirmation)]
                assert not any(groups[i]&groups[j] for i,j in ((0,1),(0,2),(1,2)))
                assert audit['objectivity']['status']=='PASS'
                frozen=json.loads((folder/'frozen_models.json').read_text())
                if task=='Task2':
                    assert all(frozen[a]['losses']['completed_optimizer_steps']==7000 for a in spec['task2']['arms'])
                if task=='Task5':
                    assert frozen['old_fmt']['parameter_count']==frozen['fmt_all_v2']['parameter_count']
                with (folder/'per_run.csv').open() as f:rows=list(csv.DictReader(f))
                if task in spec.get('reuse_arms',{}):
                    previous=Path(spec['reuse_from'])/'shards'/task/dataset/f'seed{seed}'
                    old_audit=json.loads((previous/'input_audit.json').read_text())
                    for role in ('training','validation'):
                        assert [(r['path'],r['sha256'],r['identity']) for r in audit[role]]==[(r['path'],r['sha256'],r['identity']) for r in old_audit[role]]
                    old_test=json.loads((previous/'confirmation_input_audit.json').read_text())
                    assert [(r['path'],r['sha256'],r['identity']) for r in confirmation]==[(r['path'],r['sha256'],r['identity']) for r in old_test]
                    with np.load(previous/'predictions.npz') as prior,np.load(folder/'predictions.npz') as current:
                        assert np.array_equal(prior['labels'],current['labels'])
                with np.load(folder/'predictions.npz',allow_pickle=False) as z:
                    y=z['labels'].astype(bool)
                    h=hashlib.sha256(z['labels'].tobytes()).hexdigest()
                    if reference_hash is None:reference_hash=h
                    assert reference_hash==h
                    for row in rows:
                        a=row['arm'];p=z[f'prediction_{a}'].astype(bool)
                        tp=int(np.sum(y&p));fp=int(np.sum(~y&p));fn=int(np.sum(y&~p))
                        counts={'f1':2*tp/max(2*tp+fp+fn,1),'iou':tp/max(tp+fp+fn,1),
                                'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1)}
                        if task=='Task5':
                            counts['average_precision']=float(average_precision_score(y,z[f'score_{a}']))
                            threshold=frozen[a]['threshold']
                            assert np.array_equal(p,z[f'score_{a}']>=threshold)
                        else:
                            counts['ari']=float(adjusted_rand_score(y,p))
                            counts['nmi']=float(normalized_mutual_info_score(y,p))
                        for metric,value in counts.items():
                            if abs(value-float(row[metric]))>1e-12:
                                raise ValueError(f'{folder}/{a}/{metric}: {value} differs from {row[metric]}')
                        assert int(row['sample_count'])==len(y)
                if list(folder.rglob('*.pt')):raise ValueError('temporary checkpoint remains')
                checked.append({'task':task,'dataset':dataset,'seed':seed,'rows':len(rows)})
    expected=sum(50*(len(spec[t.lower()]['arms'])+(t=='Task5')) for t in spec['tasks'])
    assert len(checked)==150 and sum(r['rows'] for r in checked)==expected
    if 'reuse_from' in spec:
        prior=json.loads((Path(spec['reuse_from'])/'independent_audit.json').read_text())
        assert prior['status']=='PASS' and prior['metric_rows']==400
    output={'status':'PASS','shards':150,'native_metric_rows':expected,'metric_rows':400,'config_sha256':sha(args.config),
            'checks':['prediction_and_metric_hashes','identical_labels_across_seeds',
                      'independent_confusion_matrices','independent_AP_ARI_NMI',
                      'disjoint_split_paths','Task2_exact_7000_updates',
                      'Task5_identical_capacity','no_temporary_checkpoints'],
            'all_shards':checked}
    (root/'independent_audit.json').write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k!='all_shards'},indent=2))


if __name__=='__main__':main()
