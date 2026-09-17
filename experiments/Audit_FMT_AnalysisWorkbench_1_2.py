"""Audit exact window rendering data without changing a classifier or partition."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(root):
    root=Path(root);manifest=json.loads((root/'build_manifest.json').read_text(encoding='utf8'))
    for rel,expected in manifest['files'].items():
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==expected,rel
    source=Path('outputs/Other_Task4C_Saliency_1.1/package')
    original=json.loads((source/'manifest.json').read_text(encoding='utf8'))
    checked=0;worst=0.
    for r in original['records']:
        with np.load(source/r['file'],allow_pickle=False) as z:
            export=json.loads((root/'saliency/data'/f"{r['id']}.json").read_text(encoding='utf8'))
            d=export['geometry_and_scores'];n=r['count'];s=d['support']
            for name in ('geometry','normalized_geometry','gradient','smooth_gradient','local_shape_delta'):
                assert np.array_equal(d[name],z[name][:n]),(r['id'],name)
            patches=d['patches'];delta=z['patch_margin_delta']
            for i,p in enumerate(patches):
                assert np.array_equal(p['indices'],z['patch_indices'][i])
                assert p['margin_delta']==delta[i]
                assert p['probability']==z['patch_probability'][i]
                assert p['probability_delta']==z['patch_probability_delta'][i]
                line,a,b=p['indices'];assert 0<=line<n and 0<=a<b<32
                # These are the exact endpoints that the new viewer connects.
                points=np.array(d['geometry'][line])[a:b+1]
                assert np.array_equal(points[0],z['geometry'][line,a])
                assert np.array_equal(points[-1],z['geometry'][line,b])
                expected_p=1/(1+np.exp(-float(z['margin'])+p['margin_delta']))
                worst=max(worst,abs(expected_p-p['probability']))
                checked+=1
            positive=[i for i,x in enumerate(delta) if x>0]
            positive=sorted(positive,key=lambda i:-delta[i])
            assert positive==s['positive_order']
            assert s['strongest_support']==(positive[0] if positive else None)
            assert all(delta[i]>0 for i in s['positive_order'])
            assert all(delta[i]<0 for i in s['negative_order'])
    assert worst<2e-7
    old=Path('outputs/Other_FMT_AnalysisWorkbench_1.1/features')
    unchanged=[]
    for rel in ('analysis_arrays.npz','analysis_summary.json','data.js','feature_ranking.csv'):
        assert (root/'features'/rel).read_bytes()==(old/rel).read_bytes();unchanged.append(rel)
    result=dict(complete=True,version=manifest['version'],bundles=len(original['records']),
        exact_interventions_checked=checked,max_probability_formula_error=worst,
        original_scores_and_geometry_unchanged=True,all_default_windows_strictly_positive=True,
        unchanged_feature_files=unchanged,artifact_hashes_checked=len(manifest['files']))
    (root/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',default='outputs/Other_FMT_AnalysisWorkbench_1.2');audit(p.parse_args().folder)
