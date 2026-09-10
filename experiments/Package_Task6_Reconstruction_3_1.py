"""Collect scalar evidence and fixed-index path examples; never collect model files."""
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np


if __name__=="__main__":
    root=Path("outputs/mainExp_Task6_Reconstruction_3.1")
    spec=json.loads((root/"config.frozen.json").read_text())
    audit=json.loads((root/"final_audit.json").read_text())
    assert audit["passed"]
    legacy=Path(spec["legacy_benchmark_cache"])
    data={}; ids=np.array([0,2000,4000])
    for dataset in spec["datasets"]:
        with np.load(legacy/"data"/dataset/"test.npz") as f:
            data[dataset+"__truth"]=f["geometry"][ids]
        with np.load(legacy/"runs"/dataset/"fmt_all_vae"/"9110"/"test_predictions.npz") as f:
            data[dataset+"__original_fmt"]=f["prediction"][ids]
        for arm in spec["arms"]:
            with np.load(root/"runs"/dataset/arm/str(spec["seeds"][0])/"legacy_test_predictions.npz") as f:
                data[dataset+"__"+arm]=f["prediction"][ids]
    np.savez(root/"fixed_examples.npz",example_ids=ids,**data)
    status=subprocess.check_output(["sacct","-j","51717895,51717896,51717897,51717898,51717899",
        "--format=JobID,State,ExitCode,Elapsed,Submit,Start,End,NodeList","-P","-X"],text=True)
    (root/"slurm_status_final.txt").write_text(status)
    with zipfile.ZipFile(root/"metadata_report.zip","w",zipfile.ZIP_DEFLATED) as z:
        for path in root.rglob("*"):
            if path.is_file() and (path.suffix in (".json",".jsonl",".csv",".txt",".out",".err") or path.name=="fixed_examples.npz"):
                z.write(path,path.relative_to(root))
    print(root/"metadata_report.zip",(root/"metadata_report.zip").stat().st_size)
