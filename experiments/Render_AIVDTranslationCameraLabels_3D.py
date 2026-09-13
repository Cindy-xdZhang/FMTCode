"""Re-render saved observed pathlines and predictions; do not recompute classifications."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from Plot_AIVDTranslationObservers_CameraLabels_3D import render


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config/Verify_AIVDTranslationObservers_1.2.json')
    args=parser.parse_args();spec=json.loads(Path(args.config).read_text())
    root=Path(spec['source_root']);out=root/spec['output_root'];out.mkdir(parents=True,exist_ok=True)
    source=root/spec['figure_source'];trajectory=root/spec['trajectory_source']
    audit=json.loads((source/'summary.json').read_text())
    assert audit['classifier']['feature']=='aivd1w3_dft'
    assert audit['cohort_count']==7220 and audit['initial_time']==10.5
    assert sha(trajectory)==audit['source_paths_sha256']
    evidence=source/'feature_evidence.npz'
    evidence_hash=sha(evidence)
    with np.load(trajectory,allow_pickle=False) as data:paths=data['pathlines']
    with np.load(evidence,allow_pickle=False) as data:labels=data['predictions']
    assert paths.shape==(7,7220,7,97,3) and labels.shape==(7,7220)
    assert np.isfinite(paths).all()
    assert [int(np.sum(y)) for y in labels]==audit['vortex_counts']
    assert [int(np.count_nonzero(y!=labels[0])) for y in labels]==audit['changed_labels_vs_original']
    before=hashlib.sha256(labels.tobytes()).hexdigest()
    started={'version':spec['version'],'job_id':os.environ.get('SLURM_JOB_ID'),
             'node':socket.gethostname(),'device':'CPU','start_time':time.time(),
             'source_paths_sha256':audit['source_paths_sha256'],'source_evidence_sha256':evidence_hash,
             'labels_array_sha256':before,'config_sha256':sha(args.config),
             'source_manifest_sha256':sha('SOURCE_MANIFEST.sha256'),
             'scope':'annotation-only; same saved paths, labels, camera and full cohort'}
    (out/'started.json').write_text(json.dumps(started,indent=2))
    for medium in ('paper','slides'):
        render(paths,labels,np.arange(7220),audit,out/'final',
               'Re160 3D | aivd1w3_dft | t0 = 10.5',medium,pdf_collision_audit=False)
    assert hashlib.sha256(labels.tobytes()).hexdigest()==before
    assert sha(evidence)==evidence_hash
    (out/'annotation_audit.json').write_text(json.dumps({**started,'status':'PASS',
        'levels':audit['levels'],'vortex_counts':audit['vortex_counts'],
        'changed_labels_vs_original':audit['changed_labels_vs_original'],
        'annotation_changes_to_labels':0,'end_time':time.time()},indent=2))
    print('RENDER_COMPLETE: seven camera icons, original paths and labels preserved.',flush=True)


if __name__=='__main__':main()
