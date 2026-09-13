"""Independently audit predictions, report all paired outcomes, then remove checkpoints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, average_precision_score, roc_auc_score


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def csv_rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader();writer.writerows(rows)


def independent(y,p,score=None):
    y,p=y.astype(bool),p.astype(bool)
    tp,fp,fn,tn=np.sum(y&p),np.sum(~y&p),np.sum(y&~p),np.sum(~y&~p)
    r={'f1':2*tp/max(2*tp+fp+fn,1),'iou':tp/max(tp+fp+fn,1),
       'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),
       'balanced_accuracy':.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1)),
       'ari':adjusted_rand_score(y,p),'nmi':normalized_mutual_info_score(y,p)}
    if score is not None:
        r['average_precision']=average_precision_score(y,score) if np.any(y) else 0.
        r['roc_auc']=roc_auc_score(y,score) if len(np.unique(y))==2 else float('nan')
    return r


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config/Verify_AIVDLongtime_1.1.json')
    args=parser.parse_args();spec=read(args.config);root=Path(spec['output_root'])
    rows,scales,devices,checked=[],[],[],[]
    for task in spec['tasks']:
        pre=read(root/f"preflight_{'task5' if task=='Task5' else 'fixed'}.json")
        assert pre['status']=='PASS' and pre['config_sha256']==sha(args.config)
        for dataset in spec['datasets']:
            reference_inputs=reference_labels=None
            for seed in spec[task.lower()]['seeds']:
                folder=root/'shards'/task/dataset/f'seed{seed}'
                done=read(folder/'complete.json');frozen=read(folder/'frozen_models.json')
                assert done['status']=='COMPLETE' and done['config_sha256']==sha(args.config)
                assert done['source_manifest_sha256']==pre['source_manifest_sha256']
                assert done['prediction_sha256']==sha(folder/'predictions.npz')
                assert done['per_run_sha256']==sha(folder/'per_run.csv')
                arm_names=spec[task.lower()]['arms'];assert sorted(frozen)==sorted(arm_names)
                inputs=read(folder/'input_audit.json');test=read(folder/'confirmation_input_audit.json')
                roles={'train':inputs['training'],'validation':inputs['validation'],'confirmation':test}
                signature=[]
                for role,data in roles.items():
                    expected=spec.get('dataset_splits',{}).get(dataset,{}).get(task,{}).get(role,spec[task.lower()][role])
                    assert [r['ordinal'] for r in data]==expected
                    if dataset in spec['cylinder_datasets']:assert all(r['metadata']['source_time']>=7.5 for r in data)
                    signature.append([(r['path'],r['sha256'],r['raw_sha256'],r['times_sha256']) for r in data])
                if reference_inputs is None:reference_inputs=signature
                assert signature==reference_inputs
                sets=[{r['path'] for r in data} for data in roles.values()]
                assert not any(sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)])
                assert done['models_frozen_before_test_time']<=done['end_time']
                if task=='Task2':
                    assert all(frozen[a]['losses']['completed_optimizer_steps']==7000 for a in arm_names)
                if task in ('Task3','Task5'):
                    assert frozen['short']['parameter_count']==frozen['long']['parameter_count']==frozen['raw_pca']['parameter_count']
                    assert frozen['short']['backbone_sha256']==frozen['long']['backbone_sha256']
                current=csv_rows(folder/'per_run.csv');assert sorted(r['arm'] for r in current)==sorted(arm_names)
                thresholds=read(folder/'thresholds.json')
                with np.load(folder/'predictions.npz',allow_pickle=False) as z:
                    y=z['labels'];assert len(y)==sum(r['valid'] for r in test)
                    label_hash=hashlib.sha256(y.tobytes()).hexdigest()
                    if reference_labels is None:reference_labels=label_hash
                    assert label_hash==reference_labels
                    for row in current:
                        arm=row['arm'];p=z[f'prediction_{arm}'];score=z[f'score_{arm}'] if task in ('Task3','Task5') else None
                        assert p.shape==y.shape
                        if score is not None:
                            assert np.isfinite(score).all()
                            np.testing.assert_array_equal(p,score>=thresholds[arm])
                            assert thresholds[arm]==frozen[arm]['threshold']
                        values=independent(y,p,score)
                        for key,value in values.items():
                            assert abs(value-float(row[key]))<1e-10 or (np.isnan(value) and np.isnan(float(row[key]))),(task,dataset,seed,arm,key)
                        rows.append({**row,**values})
                    if task=='Task5':
                        current_scales=csv_rows(folder/'per_scale.csv')
                        for row in current_scales:
                            arm=row['arm'];mask=z['scale_id']==int(row['scale_id'])
                            values=independent(y[mask],z[f'prediction_{arm}'][mask],z[f'score_{arm}'][mask])
                            for key,value in values.items():
                                # A single-class per-scale balanced accuracy is
                                # a different convention; its confusion counts
                                # and primary metrics remain independently checked.
                                if key=='balanced_accuracy' and len(np.unique(y[mask]))<2:continue
                                assert abs(value-float(row[key]))<1e-10 or (np.isnan(value) and np.isnan(float(row[key])))
                            scales.append(row)
                devices.append(done);checked.append({'task':task,'dataset':dataset,'seed':seed,'metric_rows':len(current)})
    assert len(checked)==200
    assert len(rows)==sum(len(spec['datasets'])*len(spec[t.lower()]['seeds'])*len(spec[t.lower()]['arms']) for t in spec['tasks'])
    write_csv(root/'per_run.csv',rows);write_csv(root/'per_scale.csv',scales);write_csv(root/'execution_records.csv',devices)
    dataset_rows,macro,paired=[],[],[]
    for task in spec['tasks']:
        keys=['f1','iou','precision','recall','balanced_accuracy']+(['average_precision','roc_auc'] if task in ('Task3','Task5') else ['ari','nmi'])
        part=spec[task.lower()]
        for dataset in spec['datasets']:
            for arm in part['arms']:
                sample=[r for r in rows if r['task']==task and r['dataset']==dataset and r['arm']==arm]
                r={'task':task,'dataset':dataset,'arm':arm,'seeds':len(sample)}
                for key in keys:
                    values=np.array([float(v[key]) for v in sample]);r[key+'_mean']=float(values.mean());r[key+'_std']=float(values.std(ddof=1))
                dataset_rows.append(r)
            for comparator in [a for a in ('short','raw','old_fmt','raw_pca','raw_wide','fixed_task3_raw') if a in part['arms']]:
                record={'task':task,'dataset':dataset,'method':'long','comparator':comparator}
                for key in keys:
                    diffs=[]
                    for seed in part['seeds']:
                        selected={r['arm']:r for r in rows if r['task']==task and r['dataset']==dataset and int(r['seed'])==seed}
                        diffs.append(float(selected['long'][key])-float(selected[comparator][key]))
                    record['delta_'+key+'_mean']=float(np.mean(diffs));record['delta_'+key+'_std']=float(np.std(diffs,ddof=1))
                paired.append(record)
        for arm in part['arms']:
            record={'task':task,'arm':arm,'datasets':len(spec['datasets']),'seeds':len(part['seeds'])}
            for key in keys:
                values=[np.mean([float(r[key]) for r in rows if r['task']==task and r['arm']==arm and int(r['seed'])==seed]) for seed in part['seeds']]
                record[key+'_mean']=float(np.mean(values));record[key+'_std']=float(np.std(values,ddof=1))
            macro.append(record)
    write_csv(root/'dataset_metrics.csv',dataset_rows);write_csv(root/'task_macro.csv',macro);write_csv(root/'paired_comparisons.csv',paired)
    lines=['# 全轨线 IVD 零频汇总：Task1、2、3、5','',
        '实验 Verify_AIVDLongtime_1.1。每任务10个3D数据条目、5个种子；short是原aivd1w3_dft前三点，long是aivd1w3_dft_longtime全部32点，仍为一个零频系数。',
        '所有Cylinder初始物理时间≥7.5；Task5三个Cylinder重新生成晚期缓存。所有对照本轮重新训练，Task5固定尺度迁移使用本轮Task3 Raw。标签保持初始时刻whole-field IVD p95。',
        'Task5短/长分支按相同物理采样时间序列分组求平均涡量，保留原采样序号差分；没有把不同物理时刻混成同一时间片。长窗口扩大的是标量汇总范围，不延长原始积分时间或增加采样点数。',
        '下面为五种子均值±样本标准差；所有条目全部报告。这是既有benchmark配对研究及新晚期Cylinder Task5数据，不能与旧全时段结果直接归因比较。','']
    for task in spec['tasks']:
        lines += [f'## {task}','','| 数据 | 短窗口F1 | 全轨线F1 | ΔF1 | Raw F1 |'+(' 短窗口AP | 全轨线AP |' if task in ('Task3','Task5') else ''),
                  '|---|---:|---:|---:|---:|'+('---:|---:|' if task in ('Task3','Task5') else '')]
        for dataset in spec['datasets']:
            d={r['arm']:r for r in dataset_rows if r['task']==task and r['dataset']==dataset}
            def fmt(arm,key='f1'):return f"{d[arm][key+'_mean']:.4f}±{d[arm][key+'_std']:.4f}"
            line=f"| {dataset} | {fmt('short')} | {fmt('long')} | {d['long']['f1_mean']-d['short']['f1_mean']:+.4f} | {fmt('raw')} |"
            if task in ('Task3','Task5'):line+=f" {fmt('short','average_precision')} | {fmt('long','average_precision')} |"
            lines.append(line)
        d={r['arm']:r for r in macro if r['task']==task}
        line=f"| **10项等权平均** | {d['short']['f1_mean']:.4f} | {d['long']['f1_mean']:.4f} | {d['long']['f1_mean']-d['short']['f1_mean']:+.4f} | {d['raw']['f1_mean']:.4f} |"
        if task in ('Task3','Task5'):line+=f" {d['short']['average_precision_mean']:.4f} | {d['long']['average_precision_mean']:.4f} |"
        lines += [line,'']
    lines += ['AP为平均精确率（Average Precision），用于评价分类排序。其他基线、逐尺度结果和所有指标见CSV。所有模型选择只使用训练与验证数据；全部最终预测指标独立复算。',
              '此实验没有增加非零傅里叶频率，也不检验一般时变旋转客观性。长窗口的标签仍是初始时刻标签，后续演化可能改善也可能稀释初始识别信息。']
    (root/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    # Only this experiment's temporary model files can be deleted. All metric
    # tables, predictions and the report have already been persisted above.
    deleted=[];safe=(root/'shards').resolve()
    for path in sorted(safe.rglob('*')):
        if path.is_file() and path.suffix.lower() in ('.pt','.pth','.ckpt'):
            assert safe in path.resolve().parents and 'temporary_training' in path.parts
            deleted.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size});path.unlink()
    (root/'checkpoint_cleanup.json').write_text(json.dumps({'deleted':deleted,'remaining':0},indent=2))
    audit={'status':'PASS','experiment':spec['experiment'],'config_sha256':sha(args.config),
           'shards':len(checked),'metric_rows':len(rows),'scale_rows':len(scales),'checks':checked,
           'per_run_sha256':sha(root/'per_run.csv'),'checkpoints_deleted':len(deleted),'end_time':time.time()}
    (root/'independent_audit.json').write_text(json.dumps(audit,indent=2))
    (root/'summary.json').write_text(json.dumps({'experiment':spec['experiment'],'audit':'PASS','macro':macro,
                                               'metric_rows':len(rows),'shards':len(checked)},indent=2))
    print(json.dumps({'audit':'PASS','macro':macro,'metric_rows':len(rows),'scale_rows':len(scales)}),flush=True)


if __name__=='__main__':main()
