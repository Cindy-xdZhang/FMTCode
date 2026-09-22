"""Package the incremental data and submit single-seed FMT/Conv8 jobs."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import tarfile
from experiments.Task4C_LabelSplitQuick_1_1 import sha,write,verify_sources

ROOT=Path(__file__).resolve().parents[1]
CONFIG='config/mainExp_Task4C_CouetteTraining_2.1.json'
OUT=ROOT/'outputs/mainExp_Task4C_CouetteTraining_2.1'
DATA=ROOT/'outputs/mainExp_Task4C_CouetteDataset_2.1'


def dependencies(starts):
    todo=list(starts);done=set()
    while todo:
        name=todo.pop()
        if name in done:continue
        path=ROOT/name;assert path.exists(),name;done.add(name)
        if path.suffix!='.py':continue
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:modules=[node.module]+[node.module+'.'+a.name for a in node.names]
            for module in modules:
                if module.split('.')[0] not in ('FMT_Utils','experiments','tests'):continue
                for p in [Path(*module.split('.')).with_suffix('.py'),Path(*module.split('.'))/'__init__.py']:
                    if (ROOT/p).exists():todo.append(p.as_posix())
    return sorted(done)


def files():
    result=dependencies(['experiments/Deploy_Task4C_CouetteTraining_2_1.py','experiments/Task4C_CouetteTraining_2_1.py',
        'experiments/Build_Task4C_CouetteDataset_2_1.py','experiments/Audit_Task4C_CouetteDataset_2_1.py',
        'experiments/Prepare_Task4C_CouetteLoadIndices_2_1.py','experiments/Export_Task4C_CouetteGallery_2_1.py',
        'experiments/Serve_Task4C_BundleGallery_1_6.py','experiments/Collect_Task4C_CouetteTraining_2_1.py','tests/test_task4c_raw_curl_2_1.py','tests/test_task4c_tangent_jitter_1_1.py',
        'experiments/Verify_Task4C_IntegrationSemantics_1_1.py','experiments/Task4C_RawCurlPreview_1_1.py'])
    result += [CONFIG,'config/mainExp_Task4C_CouetteDataset_2.1.json','config/mainExp_Task4C_V2NewLabelTraining_1.1.json',
               'config/mainExp_Task4C_V2NewLabel_1.1.json','config/mainExp_Task4C_GTHeadCoverage_1.1.json',
               'docs/Task4C_couette_raw_training_execution_2.1.md','docs/Task4C_integration_semantics_audit_1.1.md',
               'experiments/Start_Task4C_BundleGallery_1_6.ps1','experiments/templates/task4c_bundle_gallery_1_6.js',
               'ibex_bash/task4c_couette_training_2p1.sh']
    return sorted(set(result))


def package():
    OUT.mkdir(parents=True,exist_ok=True)
    audit=json.loads((DATA/'independent_audit.json').read_text());assert audit['complete']
    manifest={p:sha(ROOT/p) for p in files()};write(OUT/'deployment_source_sha256.json',manifest)
    with tarfile.open(OUT/'sources.tar.gz','w:gz') as archive:
        for p in manifest:archive.add(ROOT/p,arcname=p)
        archive.add(OUT/'deployment_source_sha256.json',arcname='deployment_source_sha256.json')
    selected=[DATA/n for n in ('data_audit.json','independent_audit.json','dataset.json','selection.json')]
    for folder in ('physical','neighbors','load_indices'):
        selected.extend(p for p in (DATA/folder).rglob('*') if p.is_file())
    data_manifest={p.relative_to(ROOT).as_posix():sha(p) for p in selected};write(OUT/'data_transfer_sha256.json',data_manifest)
    with tarfile.open(OUT/'couette_data.tar.gz','w:gz',compresslevel=1) as archive:
        for p in selected:archive.add(p,arcname=p.relative_to(ROOT).as_posix())
        archive.add(OUT/'data_transfer_sha256.json',arcname='data_transfer_sha256.json')
    print(json.dumps(dict(source_files=len(manifest),data_files=len(data_manifest),data_bytes=(OUT/'couette_data.tar.gz').stat().st_size)))


def verify():
    verify_sources()
    for p,h in json.loads(Path('data_transfer_sha256.json').read_text()).items():assert sha(p)==h,p
    for test in ('test_task4c_raw_curl_2_1.py','test_task4c_tangent_jitter_1_1.py'):
        subprocess.run(['/home/zhanx0o/anaconda3/envs/deepvortex/bin/python','-m','unittest','discover','-s','tests','-p',test],check=True)
    write(OUT/'source_verification.json',dict(complete=True,all_transferred_hashes_checked=True))


def submit():
    verify();s=json.loads(Path(CONFIG).read_text());out=Path(s['output']);(out/'logs').mkdir(parents=True,exist_ok=True)
    assert not (out/'submission.json').exists()
    jobs={};previous=None
    for phase,array in [('prepare',None),('encode','0-3%2'),('pilot','0-1%2'),('train','0-7%2'),('audit',None)]:
        cmd=['sbatch','--parsable','--propagate=NONE','--kill-on-invalid-dep=yes','--cpus-per-task=4','--mem=32G','--time=03:00:00',
             '--job-name=CCR21_'+phase,'--output='+str(out/'logs/%x_%A_%a.out'),'--error='+str(out/'logs/%x_%A_%a.err')]
        if phase in ('encode','pilot','train'):cmd+=['--gres=gpu:1','--constraint=v100','--exclude=gpu213-18']
        if array:cmd+=['--array='+array]
        if previous:cmd+=['--dependency=afterok:'+previous]
        previous=subprocess.check_output(cmd+['ibex_bash/task4c_couette_training_2p1.sh',phase],text=True).strip().split(';')[0]
        assert previous.isdigit();jobs[phase]=previous
        write(out/'submission.json',dict(jobs=jobs,seed=s['seed'],epochs=s['epochs'],models=['fmt','conv8'],folds=s['active_folds']))
        print(phase,previous,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('package','verify','submit','files'));a=p.parse_args()
    if a.phase=='files':print(json.dumps(files(),indent=2))
    else:globals()[a.phase]()
