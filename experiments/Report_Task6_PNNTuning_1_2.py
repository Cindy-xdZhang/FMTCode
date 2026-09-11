"""Read saved tuning records; never load geometry, checkpoints or test arrays."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report(root,config):
    root=Path(root)
    spec=read(config)
    config_hash=digest(config)
    assert digest(root/'config.frozen.json')==config_hash
    events=[read(p) for p in (root/'events').glob('*.json')]
    summary=dict(experiment=spec['experiment'],config_sha256=config_hash,
        stages=dict(Counter(e['phase']+':'+e['status'] for e in events)),
        event_failures=[e for e in events if e['status']=='FAILED'],
        test_used_for_selection=False,screen_models=0,refine_models=0)
    rows=[]
    for phase,training in [('screen',spec['screening']),('refine',spec['training'])]:
        for path in (root/phase).glob('*/*/result.json'):
            result=read(path)
            assert not result['test_read'] and result['provenance']['config_sha256']==config_hash
            candidate=path.parent.name
            items=[result] if phase=='screen' else result['rows']
            for row in items:
                fit=read(path.parent/row['arm']/'fit.json')
                assert fit['candidate']['id']==candidate
                assert fit['train_samples']==spec['primary_train_size']
                assert fit['updates']==training['updates']
                assert fit['train_examples_exposed']==training['updates']*training['batch_size']
                best=min((r for r in fit['curve'] if r['step']>0),key=lambda r:r['validation_rmse_r'])
                assert fit['selected_step']==row['selected_step']==best['step']
                for key in ('train_rmse_r','validation_rmse_r'):
                    assert math.isclose(fit[key],row[key],rel_tol=1e-10,abs_tol=1e-12)
                    assert math.isclose(fit[key],best[key],rel_tol=1e-10,abs_tol=1e-12)
                rows.append(dict(phase=phase,dataset=result['dataset'],candidate=candidate,arm=row['arm'],
                    selected_step=row['selected_step'],train_rmse_r=row['train_rmse_r'],
                    validation_rmse_r=row['validation_rmse_r'],parameters=fit['structure']['trainable_parameters']))
                summary[phase+'_models']+=1
    if (root/'promotion.json').exists():
        assert digest(root/'promotion.json')==digest(root/'promotion.frozen.json')
        promotion=read(root/'promotion.json')
        assert not promotion['test_read'] and promotion['provenance']['config_sha256']==config_hash
        summary['promoted']=[c['id'] for c in promotion['candidates']]
        summary['screen_scores']=promotion['scores']
    if (root/'selection.json').exists():
        assert digest(root/'selection.json')==digest(root/'selection.before_test.json')
        selection=read(root/'selection.json')
        assert not selection['test_read'] and selection['provenance']['config_sha256']==config_hash
        assert summary['screen_models']==len(spec['screen_datasets'])*len(spec['candidates'])
        assert summary['refine_models']==len(spec['datasets'])*spec['promote_count']*2
        selected=selection['candidate']['id']
        chosen=[r for r in rows if r['phase']=='refine' and r['candidate']==selected]
        for arm in ('pnn_trans','raw_trans'):
            values=[r['validation_rmse_r'] for r in chosen if r['arm']==arm]
            assert len(values)==len(spec['datasets'])
            assert math.isclose(sum(values)/len(values),selection['means'][arm],rel_tol=1e-12)
        reference=read(root/'baseline_reference.json')
        assert reference['selection_sha256']==spec['baseline_selection_sha256']
        assert math.isclose(sum(r['validation_rmse_r'] for r in reference['rows'])/len(reference['rows']),
                            selection['means']['raw_frozen'],rel_tol=1e-12)
        summary.update(selected_candidate=selection['candidate'],means=selection['means'],selected_rows=chosen,
            gate_passed=selection['gate_passed'],numeric_gate_passed=selection['numeric_gate_passed'],
            beats_both_baselines=selection['beats_both_baselines'],selection_sha256=digest(root/'selection.json'))
    if rows:
        with (root/'checked_training_results.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if (root/'final_audit.json').exists():
        summary['final_audit']=read(root/'final_audit.json')
    summary['record_consistency_passed']=True
    (root/'status_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='outputs/Verify_Task6_PNNTrans_1.2')
    parser.add_argument('--config',default='config/Verify_Task6_PNNTrans_1.2.json')
    args=parser.parse_args()
    report(args.root,args.config)
