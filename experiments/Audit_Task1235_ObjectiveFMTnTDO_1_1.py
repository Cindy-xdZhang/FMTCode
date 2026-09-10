"""Independent local audit from predictions; never loads training checkpoints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, average_precision_score, jaccard_score


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='outputs/Verify_Task1235_ObjectiveFMTnTDO_1.1')
    args=p.parse_args();root=Path(args.root)
    config=json.loads(Path('config/Verify_Task1235_ObjectiveFMTnTDO_1.1.json').read_text())
    remote=json.loads((root/'final_audit.json').read_text())
    assert remote['status']=='PASS'
    assert digest(root/'summary.csv')==remote['summary_sha256']
    rows=[];dataset_rows=[];seed_rows=[];checks=[]
    for task in config['tasks']:
        part=config[task.lower()]
        for dataset in config['datasets']:
            for seed in part['seeds']:
                directory=root/'shards'/task/dataset/f'seed{seed}'
                complete=json.loads((directory/'complete.json').read_text())
                thresholds=json.loads((directory/'thresholds.json').read_text())
                reported=list(csv.DictReader((directory/'per_run.csv').open()))
                assert digest(directory/'per_run.csv')==complete['per_run_sha256']
                assert digest(directory/'predictions.npz')==complete['prediction_sha256']
                assert len(reported)==len(part['arms'])
                assert {r['arm'] for r in reported}==set(part['arms'])
                with np.load(directory/'predictions.npz',allow_pickle=False) as z:
                    y=z['labels']
                    for row in reported:
                        arm=row['arm'];prediction=z['prediction_'+arm]
                        values={'f1':f1_score(y,prediction,zero_division=0),
                                'iou':jaccard_score(y,prediction,zero_division=0)}
                        if 'score_'+arm in z:
                            score=z['score_'+arm]
                            np.testing.assert_array_equal(prediction,score>=thresholds[arm])
                            values['average_precision']=average_precision_score(y,score)
                        for metric,value in values.items():
                            assert abs(float(row[metric])-value)<1e-10,(directory,arm,metric)
                        rows.append({'task':task,'dataset':dataset,'seed':seed,'arm':arm,**values})
                checks.append({'task':task,'dataset':dataset,'seed':seed,
                               'git_commit':complete['git_commit'],'node':complete['node'],'device':complete['device']})
        for arm in part['arms']:
            for dataset in config['datasets']:
                values=[r for r in rows if r['task']==task and r['dataset']==dataset and r['arm']==arm]
                dataset_rows.append({'task':task,'dataset':dataset,'arm':arm,
                    'f1_mean':float(np.mean([r['f1'] for r in values])),
                    'f1_seed_std':float(np.std([r['f1'] for r in values],ddof=1))})
            for seed in part['seeds']:
                values=[r for r in rows if r['task']==task and r['seed']==seed and r['arm']==arm]
                seed_rows.append({'task':task,'seed':seed,'arm':arm,
                    'macro_f1':float(np.mean([r['f1'] for r in values]))})
    summary=list(csv.DictReader((root/'summary.csv').open()))
    for row in summary:
        values=[r['f1'] for r in rows if r['task']==row['task'] and r['arm']==row['arm']]
        assert len(values)==int(row['runs'])==30
        assert abs(float(row['f1_mean'])-np.mean(values))<1e-10
    for name,values in [('per_dataset.csv',dataset_rows),('per_seed_macro.csv',seed_rows)]:
        with (root/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    result={'status':'PASS','rows':len(rows),'shards':len(checks),'metrics_recomputed':['f1','iou','average_precision'],
            'source_checks':checks,'models_downloaded':False,'summary_sha256':digest(root/'summary.csv')}
    (root/'local_independent_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':'PASS','rows':len(rows),'shards':len(checks)}))


if __name__=='__main__':main()
