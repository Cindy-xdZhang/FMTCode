"""Collect scalar evidence and fixed-index path examples; never collect model files."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np


def curved_example_ids(geometry):
    """Select by ground-truth total turning only, never by a model's error."""
    delta=np.diff(geometry.astype(np.float64),axis=2)
    length=np.linalg.norm(delta,axis=-1)
    denominator=length[:,:,1:]*length[:,:,:-1]
    dot=(delta[:,:,1:]*delta[:,:,:-1]).sum(-1)
    valid=(length[:,:,1:]>1e-9)&(length[:,:,:-1]>1e-9)
    cosine=np.ones_like(dot)
    np.divide(dot,denominator,out=cosine,where=valid)
    turning=np.arccos(np.clip(cosine,-1,1)).sum(-1).mean(-1)
    order=np.argsort(turning,kind='stable')
    ids=order[np.rint(np.array([.95,.99,.999])*(len(order)-1)).astype(int)]
    return ids,turning[ids]


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--examples-only',action='store_true',help='Export fixed examples from fully evaluated first-seed pairs only')
    parser.add_argument('--curved-examples',action='store_true',help='Additional legacy-test examples at the 95th/99th/99.9th percentiles of ground-truth total turning')
    args=parser.parse_args()
    if args.curved_examples and not args.examples_only:
        parser.error('--curved-examples requires --examples-only; it never replaces the fixed main examples')
    root=Path("outputs/mainExp_Task6_Reconstruction_3.1")
    spec=json.loads((root/"config.frozen.json").read_text())
    if not args.examples_only:
        audit=json.loads((root/"final_audit.json").read_text())
        assert audit["passed"]
    legacy=Path(spec["legacy_benchmark_cache"])
    data={}; ids=np.array([0,2000,4000])
    for dataset in spec["datasets"]:
        if args.examples_only and not all((root/'runs'/dataset/arm/str(spec['seeds'][0])/'result.json').exists() for arm in spec['arms']):
            continue
        with np.load(legacy/"data"/dataset/"test.npz") as f:
            geometry=f['geometry']
            if args.curved_examples:
                ids,turning=curved_example_ids(geometry)
                data[dataset+'__ids']=ids
                data[dataset+'__mean_total_turn_radians']=turning
            data[dataset+"__truth"]=geometry[ids]
        with np.load(legacy/"runs"/dataset/"fmt_all_vae"/"9110"/"test_predictions.npz") as f:
            data[dataset+"__original_fmt"]=f["prediction"][ids]
        for arm in spec["arms"]:
            with np.load(root/"runs"/dataset/arm/str(spec["seeds"][0])/"legacy_test_predictions.npz") as f:
                data[dataset+"__"+arm]=f["prediction"][ids]
    example_path=root/('curved_examples.npz' if args.curved_examples else 'fixed_examples.partial.npz' if args.examples_only else 'fixed_examples.npz')
    if args.curved_examples:
        np.savez(example_path,**data)
        metadata=dict(experiment='Other_Task6_CurvedVisualization_1.1',
            time=datetime.datetime.now().astimezone().isoformat(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            legacy_source=str(legacy),old_seed=9110,new_seed=spec['seeds'][0],
            turning_percentiles=[95,99,99.9],selection='GT seven-line mean cumulative turning only; never prediction/error selection',
            examples={d:dict(indices=data[d+'__ids'].tolist(),mean_total_turn_radians=data[d+'__mean_total_turn_radians'].tolist())
                for d in spec['datasets'] if d+'__truth' in data})
        (root/'curved_examples.metadata.json').write_text(json.dumps(metadata,indent=2))
    else:
        np.savez(example_path,example_ids=ids,**data)
    if args.examples_only:
        print(example_path,sorted(k for k in data if k.endswith('__truth')))
        raise SystemExit(0)
    status=subprocess.check_output(["sacct","-j","51717895,51717896,51717897,51717898,51717899",
        "--format=JobID,State,ExitCode,Elapsed,Submit,Start,End,NodeList","-P","-X"],text=True)
    (root/"slurm_status_final.txt").write_text(status)
    with zipfile.ZipFile(root/"metadata_report.zip","w",zipfile.ZIP_DEFLATED) as z:
        for path in root.rglob("*"):
            if path.is_file() and (path.suffix in (".json",".jsonl",".csv",".txt",".out",".err") or path.name in ('fixed_examples.npz','curved_examples.npz')):
                z.write(path,path.relative_to(root))
    print(root/"metadata_report.zip",(root/"metadata_report.zip").stat().st_size)
