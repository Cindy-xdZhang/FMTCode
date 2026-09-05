"""Recompute all submitted metrics from saved predictions, independently.

Does not import the experiment runner, training loop, or feature extractor.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score, average_precision_score, jaccard_score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as handle: return list(csv.DictReader(handle))


def audit(config):
    spec=yaml.safe_load(Path(config).read_text())
    root=Path(spec['output_root'])
    preflight=json.loads((root/'preflight.json').read_text())
    if preflight['status']!='PASS' or preflight['config_sha256']!=sha(config):
        raise ValueError('invalid preflight')
    cases=[(c['id'],r) for c in spec['corruptions'] for r in
           ([0] if c['kind']=='clean' else spec['corruption_seeds'])]
    all_rows=[]; checked_predictions=0; checked_scales=0
    for task in spec['tasks']:
        arms=set(spec['arms'])|{'geometry_threshold'}
        if task!='Task1': arms.add('raw_backbone')
        for dataset in spec['datasets']:
            dataset_labels=None; dataset_identities=None
            for seed in spec['seeds']:
                directory=root/'shards'/task/dataset/f'seed{seed}'
                complete=json.loads((directory/'complete.json').read_text())
                if complete['status']!='COMPLETE' or complete['config_sha256']!=sha(config):
                    raise ValueError('incomplete or mismatched shard')
                if sha(directory/'per_run.csv')!=complete['per_run_sha256']:
                    raise ValueError('per-run file was modified')
                if list((directory/'temporary_training').rglob('*.pt')):
                    raise ValueError('temporary model files were not removed')
                frozen=json.loads((directory/'frozen_clean_models.json').read_text())
                if task!='Task1':
                    for field in ['parameter_count','trainable_parameter_count','backbone_sha256']:
                        if len({str(frozen[a][field]) for a in spec['arms']})!=1:
                            raise ValueError(f'capacity/backbone mismatch: {field}')
                rows=read_csv(directory/'per_run.csv')
                expected={(a,c,r) for a in arms for c,r in cases}
                keyed={(r['arm'],r['condition'],int(r['corruption_seed'])):r for r in rows}
                if set(keyed)!=expected or len(keyed)!=len(rows):
                    raise ValueError('missing, duplicate, or unexpected metric rows')
                hashes={str(Path(item['path'])):item['sha256'] for item in complete['predictions']}
                scale_rows=read_csv(directory/'per_scale.csv') if task=='Task5' else []
                scale_keyed={(r['arm'],r['condition'],int(r['corruption_seed']),int(r['scale_id'])):r for r in scale_rows}
                clean_scores={}; clean_scales={}
                for condition,repeat in cases:
                    path=directory/f'predictions_{condition}_{repeat}.npz'
                    if sha(path)!=hashes[str(path)]: raise ValueError('prediction checksum mismatch')
                    with np.load(path,allow_pickle=False) as data:
                        labels=data['labels']; scores=data['scores']; thresholds=data['thresholds']
                        names=data['arms'].tolist(); scales=data['scale_ids']
                        fingerprints=json.loads(str(data['fingerprints_json']))
                    identities=[v['identity'] for v in fingerprints]
                    if dataset_labels is None:
                        dataset_labels=labels.copy(); dataset_identities=identities
                    if not np.array_equal(labels,dataset_labels) or identities!=dataset_identities:
                        raise ValueError('sample identities/labels differ across conditions or training seeds')
                    if set(names)!=arms or scores.shape!=(len(arms),len(labels)) or not np.isfinite(scores).all():
                        raise ValueError('invalid prediction matrix')
                    for i,arm in enumerate(names):
                        row=keyed[(arm,condition,repeat)]
                        expected_threshold=0.0 if task=='Task1' and arm!='geometry_threshold' else frozen[arm]['threshold']
                        if float(thresholds[i])!=float(expected_threshold) or float(row['threshold'])!=float(expected_threshold):
                            raise ValueError('threshold changed after clean freeze')
                        groups=[(None,np.ones(len(labels),dtype=bool))]
                        if task=='Task5': groups.extend((int(s),scales==s) for s in np.unique(scales))
                        for scale,mask in groups:
                            actual=row if scale is None else scale_keyed[(arm,condition,repeat,scale)]
                            y=labels[mask]; p=scores[i,mask]; decision=p>=thresholds[i]
                            values={'f1':f1_score(y,decision,zero_division=0),
                                    'average_precision':average_precision_score(y,p) if y.any() else 0.0,
                                    'iou':jaccard_score(y,decision,zero_division=0)}
                            if int(actual['sample_count'])!=int(mask.sum()): raise ValueError('sample count mismatch')
                            key=(arm,scale)
                            if condition=='clean': clean_scores[key]=values
                            for metric,value in values.items():
                                if abs(float(actual[metric])-float(value))>1e-8:
                                    raise ValueError(f'metric mismatch: {path}/{arm}/{scale}/{metric}')
                                drop=clean_scores[key][metric]-value
                                if abs(float(actual[metric+'_drop_from_clean'])-float(drop))>1e-8:
                                    raise ValueError('incorrect degradation relative to paired clean baseline')
                            if scale is not None: checked_scales+=1
                        checked_predictions+=1
                if task=='Task5' and len(scale_rows)!=len(expected)*len(np.unique(scales)):
                    raise ValueError('incomplete scale table')
                all_rows.extend(rows)
    # Recompute every dataset-equal aggregate from the audited raw rows.
    table=read_csv(root/'robustness_table.csv')
    keys={(r['task'],r['condition'],r['arm']) for r in all_rows}
    if {(r['task'],r['condition'],r['arm']) for r in table}!=keys or len(table)!=len(keys):
        raise ValueError('incomplete summary table')
    for row in table:
        subset=[r for r in all_rows if all(r[k]==row[k] for k in ('task','condition','arm'))]
        for metric in ['f1','average_precision','iou','f1_drop_from_clean','average_precision_drop_from_clean','iou_drop_from_clean']:
            value=np.mean([np.mean([float(r[metric]) for r in subset if r['dataset']==d]) for d in spec['datasets']])
            if abs(float(row[metric])-float(value))>1e-8: raise ValueError('incorrect macro summary')
    result={'status':'PASS','config_sha256':sha(config),'checked_predictions':checked_predictions,
            'checked_scale_rows':checked_scales,'rows':len(all_rows),'table_sha256':sha(root/'robustness_table.csv')}
    (root/'independent_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config/Verify_Task135_GeometricControls_1.1.yaml')
    audit(p.parse_args().config)
