"""Verify completeness and summarize paired encoder reruns without selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config/Verify_FMTAllV2_1.1.json')
    parser.add_argument('--root')
    args=parser.parse_args()
    config=Path(args.config);spec=json.loads(config.read_text())
    root=Path(args.root or spec['output_root'])
    rows=[];scale_rows=[];devices=[];invariance=[];parameters=[]
    for task in spec['tasks']:
        expected=spec[task.lower()]['arms']+(['raw_backbone'] if task=='Task5' else [])
        for dataset in spec['datasets']:
            for seed in spec[task.lower()]['seeds']:
                folder=root/'shards'/task/dataset/f'seed{seed}'
                complete=json.loads((folder/'complete.json').read_text())
                if complete['config_sha256']!=digest(config):raise ValueError('changed config')
                if digest(folder/'per_run.csv')!=complete['per_run_sha256']:raise ValueError('changed per-run metrics')
                if complete['temporary_checkpoints_remaining']!=0:raise ValueError('checkpoints remain')
                with (folder/'per_run.csv').open() as f:current=list(csv.DictReader(f))
                assert sorted(r['arm'] for r in current)==sorted(expected)
                assert all(r['task']==task and r['dataset']==dataset and int(r['seed'])==seed for r in current)
                assert len({int(r['sample_count']) for r in current})==1
                for r in current:
                    for key in ('f1','iou','precision','recall'):
                        assert np.isfinite(float(r[key])) and 0<=float(r[key])<=1
                rows.extend(current)
                for arm in spec.get('reuse_arms',{}).get(task,[]):
                    previous=Path(spec['reuse_from'])/'shards'/task/dataset/f'seed{seed}'
                    source_done=json.loads((previous/'complete.json').read_text())
                    assert digest(previous/'per_run.csv')==source_done['per_run_sha256']
                    with (previous/'per_run.csv').open() as f:
                        reused=[r for r in csv.DictReader(f) if r['arm']==arm]
                    assert len(reused)==1
                    rows.append({**reused[0],'reused_from':str(previous)})
                if task=='Task5':
                    with (folder/'per_scale.csv').open() as f:sub=list(csv.DictReader(f))
                    assert len(sub)==27 and len({r['scale_id'] for r in sub})==9
                    scale_rows.extend(sub)
                devices.append(complete)
                audit=json.loads((folder/'input_audit.json').read_text())
                invariance.append({'task':task,'dataset':dataset,'seed':seed,**audit['objectivity']})
                frozen=json.loads((folder/'frozen_models.json').read_text())
                for arm in spec[task.lower()]['arms']:
                    entry=frozen[arm]
                    count=entry.get('parameter_count',entry.get('losses',{}).get('parameter_count',0))
                    parameters.append({'task':task,'dataset':dataset,'seed':seed,'arm':arm,'parameters':count,
                                       'best_epoch':entry.get('best_epoch',''),
                                       'updates':entry.get('losses',{}).get('completed_optimizer_steps','')})
    table=[];macro=[]
    for task in spec['tasks']:
        keys=['f1','iou','precision','recall']+(['average_precision'] if task=='Task5' else ['ari','nmi'])
        for dataset in spec['datasets']:
            subset=[r for r in rows if r['task']==task and r['dataset']==dataset]
            result={'task':task,'dataset':dataset}
            for key in keys:
                for arm in sorted({r['arm'] for r in subset}):
                    values=[float(r[key]) for r in subset if r['arm']==arm]
                    result[f'{arm}_{key}_mean']=float(np.mean(values))
                    result[f'{arm}_{key}_std']=float(np.std(values,ddof=1))
                delta=[next(float(r[key]) for r in subset if r['arm']=='fmt_all_v2' and int(r['seed'])==s)
                       -next(float(r[key]) for r in subset if r['arm']=='old_fmt' and int(r['seed'])==s)
                       for s in spec[task.lower()]['seeds']]
                result[f'delta_{key}_mean']=float(np.mean(delta))
                result[f'delta_{key}_std']=float(np.std(delta,ddof=1))
            table.append(result)
        current=[r for r in table if r['task']==task]
        result={'task':task,'datasets':len(current),'seeds_per_dataset':5,
                'datasets_improved_f1':sum(r['delta_f1_mean']>0 for r in current),
                'datasets_decreased_f1':sum(r['delta_f1_mean']<0 for r in current)}
        for field in current[0]:
            if field.endswith('_mean'):result[field]=float(np.mean([r[field] for r in current]))
        # Variation of the equal-dataset macro over paired run indices.
        for key in keys:
            values=[]
            for s in spec[task.lower()]['seeds']:
                subset=[r for r in rows if r['task']==task and int(r['seed'])==s]
                values.append(np.mean([float(r[key]) for r in subset if r['arm']=='fmt_all_v2'])-
                              np.mean([float(r[key]) for r in subset if r['arm']=='old_fmt']))
            result[f'delta_{key}_macro_seed_std']=float(np.std(values,ddof=1))
        macro.append(result)
    write_csv(root/'per_run.csv',rows)
    write_csv(root/'per_scale.csv',scale_rows)
    write_csv(root/'dataset_comparison.csv',table)
    write_csv(root/'task_macro.csv',macro)
    write_csv(root/'execution_records.csv',devices)
    write_csv(root/'parameter_counts.csv',parameters)
    write_csv(root/'objectivity_checks.csv',invariance)
    summary={'status':'COMPLETE','experiment':spec['experiment'],'config_sha256':digest(config),
             'shards':len(devices),'per_run_rows':len(rows),'per_scale_rows':len(scale_rows),
             'max_objectivity_absolute_error':max(r['max_absolute_error'] for r in invariance),
             'max_objectivity_relative_l2_error':max(r['relative_l2_error'] for r in invariance),
             'macro':macro,'dataset_comparison':table}
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=[f"# {spec['experiment']} — paired frozen-benchmark results",'',
           'F1 is shown on a 0–1 scale; differences are new minus old. Each dataset has five paired seeds. Macro gives each of ten datasets equal weight. These are previously used benchmarks, not a fresh confirmation.','',
           '| Dataset | Task1 old → v2 (Δ) | Task2 old → v2 (Δ) | Task5 old → v2 (Δ) |',
           '|---|---:|---:|---:|']
    for dataset in spec['datasets']:
        cells=[]
        for task in spec['tasks']:
            r=next(r for r in table if r['task']==task and r['dataset']==dataset)
            cells.append(f"{r['old_fmt_f1_mean']:.4f} → {r['fmt_all_v2_f1_mean']:.4f} ({r['delta_f1_mean']:+.4f})")
        lines.append('| '+dataset+' | '+' | '.join(cells)+' |')
    lines.append('| **Macro** | '+' | '.join(f"{r['old_fmt_f1_mean']:.4f} → {r['fmt_all_v2_f1_mean']:.4f} ({r['delta_f1_mean']:+.4f})" for r in macro)+' |')
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='dataset_comparison'},indent=2))


if __name__=='__main__':main()
