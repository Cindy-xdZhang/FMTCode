"""Independent cohort and label integrity audit; no models or quality scores."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from FMT_Utils.LargeNeighbor_3D import large_neighbor_fmt
from experiments.Run_Task135_GeometricControls import write_csv,write_json,sha


def audit(spec,config_path):
    root=Path(spec['output_root']);rows=[]
    for path in sorted((root/'cache').glob('*/*/slice_*.npz')):
        metadata=json.loads(path.with_suffix('.json').read_text())
        assert metadata['cache_sha256']==sha(path) and metadata['config_sha256']==sha(config_path)
        original=Path(metadata['old_cache_path']);assert sha(original)==metadata['old_cache_sha256']
        with np.load(path,allow_pickle=False) as a,np.load(original,allow_pickle=False) as old:
            ids=a['retained_indices'];target=a['labels'];raw=a['raw25'];n=len(target)
            assert np.all(np.diff(ids)>0) and len(ids)==len(raw)
            np.testing.assert_array_equal(target,old['reference'][ids])
            np.testing.assert_array_equal(a['raw7'],old['raw_features'][ids].reshape(n,7,32,3))
            np.testing.assert_array_equal(a['seeds'],old['seeds'][ids])
            assert (a['line_lengths'][ids]==49).all()
            distance=np.linalg.norm(raw[:,1:,0]-raw[:,:1,0],axis=-1)
            expected=np.broadcast_to(np.repeat(np.array([1.,1.5,2.])*metadata['radius'],8),(n,24))
            np.testing.assert_allclose(distance,expected,rtol=2e-4,atol=1e-6)
            # Only a training-source subset is used to recalculate descriptors.
            is_training=(metadata['phase']=='development' and metadata['ordinal'] in spec.get('dataset_splits',{}).get(metadata['dataset'],{}).get('Task1',{}).get('train',spec['task1']['train']))
            if is_training:np.testing.assert_array_equal(a['large'][:16],large_neighbor_fmt(raw[:16]))
            rows.append({'dataset':metadata['dataset'],'phase':metadata['phase'],'ordinal':metadata['ordinal'],
                         'initial_time':metadata['source_time'],'old_valid_count':len(old['reference']),
                         'common_valid_count':n,'newly_excluded':len(old['reference'])-n,
                         'common_positive_count':int(target.sum()),'common_positive_fraction':float(target.mean()),
                         'original_positive_count':int(old['reference'].sum()),'cache_sha256':metadata['cache_sha256'],
                         'cached_training_feature_recomputed':is_training})
    expected=sum(len(json.loads(p.read_text())['slices']) for p in (root/'build_audits').glob('*.json') if not p.name.endswith('_smoke.json'))
    assert len(rows)==expected and len({r['dataset'] for r in rows})==len(spec['datasets'])
    write_csv(root/'sampling_audit.csv',rows)
    result={'status':'PASS','files':len(rows),'config_sha256':sha(config_path),'auditor_sha256':sha(__file__),
            'labels_exactly_match_original_common_subset':True,'original_seeds_and_seven_lines_exact':True,
            'all_25_lines_complete':True,'three_radii_verified':True,
            'cached_training_features_recomputed':True,
            'metadata_note':'Cache metadata inherited ivd_positive_count/fraction describe original source cohort; sampling_audit.csv gives explicit common-cohort counts.',
            'common_samples':sum(r['common_valid_count'] for r in rows),'newly_excluded':sum(r['newly_excluded'] for r in rows)}
    write_json(root/'cache_integrity_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/Verify_LargeNeighbor_1.1.json')
    args=p.parse_args();audit(json.loads(Path(args.config).read_text()),args.config)
