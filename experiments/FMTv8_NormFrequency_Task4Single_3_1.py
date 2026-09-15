"""Scheduling adapter: one frozen Task4-c candidate per short V100 allocation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# The batch working directory is the immutable c61434ac source checkout.
sys.path.insert(0,str(Path.cwd()))
import torch
from experiments import FMTv8_NormFrequency_3_1 as core


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--config',default=core.DEFAULT_CONFIG);args=parser.parse_args()
    spec=core.load_spec(args.config);core.execution_settings()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0)
    folder=Path(__file__).resolve().parent
    manifest=json.loads((folder/'ADAPTER_MANIFEST.json').read_text())
    for name,value in manifest.items():assert core.old.sha(folder/name)==value
    assert 0<=args.index<20
    checks=list((Path(spec['output'])/'control_checks').glob('*.json'))
    assert len(checks)==21 and all(json.loads(p.read_text())['status']=='PASS' for p in checks)
    assert json.loads((Path(spec['output'])/'preflight_r2.json').read_text())['status']=='PASS'
    definition=spec['candidates'][args.index]
    data={role:core.read_cache(spec,'Task4C','channel_tbl',role) for role in ('train','validation')}
    destination=core.folder_for(spec,'search','Task4C','channel_tbl',definition['id'])
    result,item=core.task4_fit(spec,args.config,spec['screen_task4_seed'],definition,'h0',data,destination)
    del item
    result['execution_adapter']=dict(candidate_index=args.index,candidate_id=definition['id'],
        adapter_sha256=core.old.sha(__file__),manifest_sha256=core.old.sha(folder/'ADAPTER_MANIFEST.json'),
        adapter_commit=(folder/'ADAPTER_COMMIT.txt').read_text().strip(),
        description='One candidate per allocation; frozen c61434ac task4_fit unchanged')
    core.write(destination/'result.json',result);core.audit_validation(destination)
    core.cleanup_weights(spec,destination)
    print('FINISHED',definition['id'],result['validation']['f1'],flush=True)


if __name__=='__main__':main()
