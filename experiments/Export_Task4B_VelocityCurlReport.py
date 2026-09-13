"""Export canonical class names from saved predictions without changing evidence.

The reused historical metrics helper spells class 3 ``hairpin_limb``. Task4-b
4.1 defines class 3 as ``hairpin_leg``. This exporter reconstructs all named
metrics from numeric predictions; it preserves the original run artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from FMT_Utils.Task4B_VelocityCurlLabels_3D import CLASS_NAMES


def export(output):
    output=Path(output)
    build=json.loads((output/'build_summary.json').read_text())
    fit=json.loads((output/'fit_summary.json').read_text())
    audit=json.loads((output/'independent_audit.json').read_text())
    pred_path=output/'predictions'/f"fmt_only_seed{fit['pooled']['seed']}.npz"
    prediction_hash=hashlib.sha256(pred_path.read_bytes()).hexdigest()
    if prediction_hash!=fit['prediction_sha256'] or fit['classes']!=list(CLASS_NAMES):
        raise RuntimeError('Prediction hash or declared class order differs')
    metrics={}
    with np.load(output/'cache.npz') as c, np.load(pred_path) as p:
        truth=c['labels'].astype(int)
        if not np.array_equal(truth,p['targets']):
            raise RuntimeError('Prediction targets differ from cache')
        pred=p['logits'].argmax(1)
        for name,mask in [('pooled',np.ones(len(truth),bool)),('channel',c['volume_codes']==0),('tbl',c['volume_codes']==1)]:
            cm=np.bincount(4*truth[mask]+pred[mask],minlength=16).reshape(4,4)
            observed=fit['pooled'] if name=='pooled' else fit['per_volume'][name]
            if cm.tolist()!=observed['confusion_matrix_true_rows_predicted_columns']:
                raise RuntimeError(f'{name} confusion matrix mismatch')
            per_class={}
            for k,label in enumerate(CLASS_NAMES):
                tp=int(cm[k,k]); support=int(cm[k].sum()); predicted=int(cm[:,k].sum())
                per_class[label]=dict(support=support,precision=tp/predicted if predicted else 0.,
                    recall=tp/support if support else 0.,f1=2*tp/(support+predicted) if support+predicted else 0.)
            metrics[name]=dict(sample_count=int(mask.sum()),error_count=int(np.sum(truth[mask]!=pred[mask])),
                accuracy=float(np.mean(truth[mask]==pred[mask])),macro_f1=float(np.mean([m['f1'] for m in per_class.values()])),
                per_class=per_class,confusion_matrix_true_rows_predicted_columns=cm.tolist())
    report=dict(experiment=build['experiment'],classes=list(CLASS_NAMES),variant='fmt_only',shared_model=True,
        seed=fit['pooled']['seed'],epochs=fit['pooled']['epochs_executed'],parameter_count=fit['pooled']['parameter_count'],
        passed=fit['pooled']['passed'] and audit['status']=='PASS' and audit['memorization_pass'],
        metrics=metrics,thresholds={name:dict(**r['thresholds'],vortex_fraction=r['full_grid_vortex_fraction']) for name,r in build['flows'].items()},
        limitations=dict(fit_equals_evaluation=True,invalid_primitives={name:r['invalid_primitive_count'] for name,r in build['flows'].items()},
            missing_valid_gt_ids={name:sorted(set(r['original_gt_ids'])-set(r['valid_gt_ids'])) for name,r in build['flows'].items()}),
        original_prediction_sha256=prediction_hash,
        metric_key_note='Original historical helper key hairpin_limb refers to numeric class 3; this version defines that class as hairpin_leg. No labels or numeric results changed.')
    path=output/'four_class_report.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(path)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/Verify_Task4B_VelocityCurlMemorization_4.1')
    export(parser.parse_args().output)
