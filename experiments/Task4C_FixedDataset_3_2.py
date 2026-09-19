"""Build class-balanced v3: 40M proposals, 8M Poisson seeds, about 2:1 after cleaning."""
import argparse
import json
import shutil
from pathlib import Path
from types import FunctionType
import numpy as np
from scipy.spatial import cKDTree
from experiments import Task4C_FixedDataset_3_1 as previous
from FMT_Utils import Task4C_FixedDataset_3_2 as balanced

CONFIG='config/mainExp_Task4C_FixedDataset_3.2.json'
FILES=previous.FILES+('FMT_Utils/Task4C_FixedDataset_3_2.py','experiments/Task4C_FixedDataset_3_2.py',
    'tests/test_task4c_fixed_dataset_3_2.py','ibex_bash/task4c_fixed_dataset_3p2.sh',
    'docs/Task4C_fixed_dataset_protocol_3.2.md',CONFIG)
sha,write=previous.sha,previous.write


def identity(config=CONFIG):
    return FunctionType(previous.identity.__code__,dict(previous.__dict__,FILES=FILES))(config)


def validate_spec(spec):
    old=json.loads(Path(previous.CONFIG).read_text())
    for key in ('physical_config','flows','integration','folds','audit','input_roots','reference_folds',
                'reference_dataset','reference_audit_sha256','pilot'):
        assert spec[key]==old[key],key
    for key,value in old['sampling'].items():
        if key!='rule':assert spec['sampling'][key]==value,key
    assert spec['balance']==dict(positive_to_negative_ratio=2.,ratio_relative_tolerance=.05,
        cleaning_correction='pilot_survival_per_class',poisson='single_global_radius_with_class_quotas')
    assert spec['output']=='outputs/mainExp_Task4C_FixedDataset_3.2'


def check(path,config):
    r=json.loads(Path(path).read_text());assert r['complete'] and r['identity']==identity(config)
    return r


def verify_inputs(spec,config):
    validate_spec(spec);reference=Path(spec['reference_dataset'])
    assert sha(reference/'data_audit.json')==spec['reference_audit_sha256']
    for flow in spec['flows']:
        name=flow['name'];p=json.loads((reference/'physical'/name/'preparation.json').read_text())
        for key in ('fold_of_instance','groups'):assert spec['reference_folds'][name][key]==p['folds'][key]
    root=Path(spec['output']);root.mkdir(parents=True,exist_ok=True);disk=shutil.disk_usage(root)
    assert disk.free>=40*2**30
    fn=previous.previous.verify_inputs
    FunctionType(fn.__code__,dict(previous.previous.__dict__,identity=identity))(spec,config)
    write(root/'resource_check.json',dict(complete=True,identity=identity(config),free_bytes=disk.free,formal_fits=0,gpu_jobs=0))


def pilot_check(spec,config,index):
    folder=Path(spec['output'])/'pilot'/spec['flows'][index]['name'];prep=check(folder/'preparation.json',config)
    for file,digest in prep['files'].items():assert sha(folder/file)==digest
    data=balanced.adapter(spec,index,True);scene=data.load_scene(spec,index);seeds,curves,meta=data.load(folder)
    sampling=dict(spec['sampling'],**spec['pilot']);mask=data.candidate_mask(scene['lambda2'],scene['oyf'],scene['threshold'])
    cells=data.candidate_cells(mask)
    pool,_=data.initial_pool(scene['axes'],scene['lambda2'],scene['oyf'],scene['threshold'],cells,
        sampling['pool_per_flow'],sampling['pool_seed']+index,sampling['pool_chunk'])
    order=np.random.default_rng([sampling['poisson_seed'],index]).permutation(len(pool))
    selected,radius,log=data.poisson_disk(pool[order],sampling['samples_per_flow'],prep['poisson']['radius_guess'],
        sampling['radius_relative_tolerance'],sampling['grid_capacity_per_cell'])
    assert radius==prep['poisson']['radius'] and log==prep['poisson']['bisection']
    np.testing.assert_array_equal(pool[meta['pool_index']],seeds)
    assert np.isin(meta['pool_index'],order[selected]).all()
    owner,_=data.sample_gt(scene['gt'],pool[order[selected]],scene['locator'])
    q=balanced.quotas(sampling['samples_per_flow'],balanced.fraction(spec,index,True))
    np.testing.assert_array_equal(np.bincount((owner>=0).astype(int),minlength=2),q)
    # Independent whole-set spatial check, including pairs of opposite labels.
    assert (cKDTree(pool[order[selected]]).query(pool[order[selected]],k=2)[0][:,1]>=radius*(1-1e-12)).all()
    owner,_=data.sample_gt(scene['gt'],seeds,scene['locator'])
    np.testing.assert_array_equal(owner,meta['gt_owner']);np.testing.assert_array_equal(owner>=0,meta['label'])
    mapping=spec['reference_folds'][spec['flows'][index]['name']]['fold_of_instance']
    assert prep['folds']['fold_of_instance']==mapping
    np.testing.assert_array_equal(meta['fold'],[mapping[str(i)] for i in meta['instance']])
    sub=np.sort(np.random.default_rng([spec['audit']['seed'],index]).choice(len(seeds),min(500,len(seeds)),replace=False))
    re=data.trace_curves(scene['grid'],seeds[sub],scene['flow_spec']['ds'],scene['steps'],spec['integration'])
    assert re['valid'].all();np.testing.assert_allclose(re['curves'],curves[sub],atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(re['arcs'],meta['half_arc_lengths'][sub]);np.testing.assert_array_equal(re['termination'],meta['half_termination'][sub])
    kept=np.bincount(meta['label'],minlength=2)
    np.testing.assert_array_equal(kept,prep['balance']['kept_by_class'])
    write(folder/'pilot_check.json',dict(complete=True,identity=identity(config),samples=len(seeds),selected_by_class=q.tolist(),
        kept_by_class=kept.tolist(),pool_and_poisson_reproduced=True,global_cross_class_separation=True,
        exact_v2_instance_fold_map=True,reintegrated=len(sub)))
    print(json.dumps(dict(flow=spec['flows'][index]['name'],pilot_check='PASS',selected=q.tolist(),kept=kept.tolist())),flush=True)


def calibrate(spec,config):
    root=Path(spec['output']);flows={}
    for flow in spec['flows']:
        name=flow['name'];path=root/'pilot'/name/'pilot_check.json';r=check(path,config)
        p,survival=balanced.corrected_fraction(r['selected_by_class'],r['kept_by_class'],spec['balance']['positive_to_negative_ratio'])
        flows[name]=dict(positive_fraction=p,pilot_sha256=sha(path),survival_by_class=survival,
            formal_pool_quotas=balanced.quotas(spec['sampling']['pool_per_flow'],p).tolist(),
            formal_sample_quotas=balanced.quotas(spec['sampling']['samples_per_flow'],p).tolist())
    assert not (root/'calibration.json').exists()
    write(root/'calibration.json',dict(complete=True,identity=identity(config),flows=flows,target_ratio=2.,
        rule='p = target_ratio * negative_survival / (positive_survival + target_ratio * negative_survival)'))
    print(json.dumps(flows),flush=True)


def audit_data(spec,config):
    root=Path(spec['output']);check(root/'calibration.json',config)
    for flow in spec['flows']:
        name=flow['name'];prep=check(root/'physical'/name/'preparation.json',config)
        assert prep['folds']['fold_of_instance']==spec['reference_folds'][name]['fold_of_instance']
    fn=previous.previous.audit_data
    collected={}
    FunctionType(fn.__code__,dict(previous.previous.__dict__,data=balanced.adapter(spec),identity=identity,
        write=lambda path,report:collected.update(report)))(spec,config)
    audit=collected
    for index,flow in enumerate(spec['flows']):
        name=flow['name'];folder=root/'physical'/name;prep=json.loads((folder/'preparation.json').read_text())
        with np.load(folder/'metadata.npz') as z:labels=z['label'];fold=z['fold']
        ratio=float(labels.sum()/(len(labels)-labels.sum()))
        assert abs(ratio/2-1)<=spec['balance']['ratio_relative_tolerance']
        np.testing.assert_array_equal(np.bincount(labels,minlength=2),prep['balance']['kept_by_class'])
        audit['flows'][name]['balance']=dict(prep['balance'],per_fold={str(f):np.bincount(labels[fold==f],minlength=2).tolist() for f in range(5)})
    audit.update(rules=spec['rules'],exact_v2_instance_fold_map=True,calibration_sha256=sha(root/'calibration.json'),
        target_positive_to_negative_ratio=2.,ratio_relative_tolerance=spec['balance']['ratio_relative_tolerance'])
    write(root/'data_audit.json',audit)


def runtime(spec,config,phase,state,code):
    return FunctionType(previous.runtime.__code__,dict(previous.__dict__,identity=identity))(spec,config,phase,state,code)


def submit(spec,config):
    import subprocess
    from datetime import datetime,timezone
    validate_spec(spec);root=Path(spec['output']);(root/'logs').mkdir(parents=True,exist_ok=True)
    assert not (root/'submission.json').exists(),'Already submitted'
    jobs={}
    phases=[('verify-inputs',[],None,'00:30:00','16G'),('pilot',['verify-inputs'],'0-1%2','03:00:00','48G'),
        ('calibrate',['pilot'],None,'00:20:00','4G'),('build',['calibrate'],'0-1%2','24:00:00','96G'),
        ('audit-data',['build'],None,'12:00:00','96G')]
    for phase,deps,array,wall,memory in phases:
        cmd=['sbatch','--parsable','--propagate=NONE','--partition=batch','--job-name=FixedDS32_'+phase,
            '--cpus-per-task=4','--mem='+memory,'--time='+wall,'--kill-on-invalid-dep=yes',
            '--output='+str(root/'logs/%x_%A_%a.out'),'--error='+str(root/'logs/%x_%A_%a.err')]
        if deps:cmd+=['--dependency=afterok:'+':'.join(jobs[d] for d in deps)]
        if array:cmd+=['--array='+array]
        cmd+=['ibex_bash/task4c_fixed_dataset_3p2.sh',phase,config]
        jobs[phase]=subprocess.check_output(cmd,text=True).strip().split(';')[0]
        previous.previous.frozen.append_locked(root/'submissions.jsonl',json.dumps(dict(phase=phase,job=jobs[phase],command=cmd,at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
        write(root/'submission.json',dict(identity=identity(config),jobs=jobs,formal_fits=0,gpu_jobs=0))
        print(phase,jobs[phase],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['verify-inputs','pilot','calibrate','build','audit-data','submit','runtime'])
    p.add_argument('--config',default=CONFIG);p.add_argument('--index',type=int,default=0)
    p.add_argument('--runtime-phase');p.add_argument('--state');p.add_argument('--exit-code',type=int)
    a=p.parse_args();spec=json.loads(Path(a.config).read_text());validate_spec(spec)
    if a.phase in ('pilot','build'):
        if a.phase=='build':check(Path(spec['output'])/'calibration.json',a.config)
        balanced.prepare(spec,a.index,write,sha,lambda:identity(a.config),pilot=a.phase=='pilot')
        if a.phase=='pilot':pilot_check(spec,a.config,a.index)
    elif a.phase=='runtime':runtime(spec,a.config,a.runtime_phase,a.state,a.exit_code)
    else:globals()[a.phase.replace('-','_')](spec,a.config)


if __name__=='__main__':main()
