"""Check CPU-cache versus V100 inference encoding on training data only."""
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
import time
import numpy as np
import torch
import sklearn
from experiments.ASAPFMT_Task35_1_1 import CONFIG, deterministic, identity, records, sha, spec_load, units, write
from FMT_Utils import ASAPFMT_Task35_1_1 as method


def main():
    spec,parent=spec_load(CONFIG);deterministic('cuda');root=Path(spec['output'])
    result=[];start=time.perf_counter()
    started_at=datetime.now(timezone.utc).isoformat()
    write(root/'device_encoding_started.json',dict(identity=identity(CONFIG),
        verifier_sha256=sha(__file__),job_id=os.environ.get('SLURM_JOB_ID'),
        started_at=started_at,node=socket.gethostname()))
    for task,dataset in units(spec):
        cache=root/'cache'/task/dataset
        manifest=json.loads((cache/'manifest.json').read_text());assert manifest['identity']==identity(CONFIG)
        norm=json.loads((cache/'normalization.json').read_text())
        r=records(parent,task,dataset,'train')[0]
        take=np.r_[np.flatnonzero(r['labels']==0)[:16],np.flatnonzero(r['labels']==1)[:16]]
        assert len(take)==32
        for arm in method.ARMS:
            original=method.encode(r['raw'][take],r['times'][take],arm,'cpu').numpy()
            changed=method.encode(r['raw'][take],r['times'][take],arm,'cuda').cpu().numpy()
            a=method.standardize(original,norm[arm]);b=method.standardize(changed,norm[arm])
            cached=np.load(cache/'train'/(arm+'.npy'),mmap_mode='r')[take]
            # This also checks actual persisted CPU features, not only a fresh call.
            cache_close=np.isclose(a,cached,atol=2e-4,rtol=2e-4)
            assert np.isfinite(original).all() and np.isfinite(changed).all() and np.isfinite(b).all()
            raw_close=np.isclose(original,changed,atol=2e-4,rtol=2e-4)
            standardized_close=np.isclose(a,b,atol=2e-3,rtol=2e-3)
            result.append(dict(task=task,dataset=dataset,arm=arm,samples=len(take),
                feature_max_absolute_error=float(np.max(np.abs(original-changed))),
                standardized_max_absolute_error=float(np.max(np.abs(a-b))),
                standardized_rms_error=float(np.sqrt(np.mean((a-b)**2))),
                cache_max_absolute_error=float(np.max(np.abs(a-cached))),
                cache_tolerance_pass=bool(cache_close.all()),cache_tolerance_failures=int((~cache_close).sum()),
                raw_tolerance_pass=bool(raw_close.all()),standardized_tolerance_pass=bool(standardized_close.all()),
                raw_tolerance_failures=int((~raw_close).sum()),standardized_tolerance_failures=int((~standardized_close).sum()),
                feature_values=int(a.size)))
            print(json.dumps(result[-1]),flush=True)
        print(task,dataset,'MEASURED',flush=True)
    write(root/'device_encoding_verification.json',dict(status='MEASURED',identity=identity(CONFIG),
        strict_tolerance_pass=all(x['raw_tolerance_pass'] and x['standardized_tolerance_pass'] and x['cache_tolerance_pass'] for x in result),
        previous_failed_jobs=['52002617','52003015'],
        original_tolerances=dict(raw_atol=2e-4,raw_rtol=2e-4,standardized_atol=2e-3,standardized_rtol=2e-3),
        verifier_sha256=sha(__file__),job_id=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),
        test_read=False,training_runs=0,seconds=time.perf_counter()-start,checks=result,
        started_at=started_at,ended_at=datetime.now(timezone.utc).isoformat(),node=socket.gethostname(),
        environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,
            sklearn=sklearn.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())))
    print('MEASURED',len(result),'checks',flush=True)


if __name__=='__main__':main()
