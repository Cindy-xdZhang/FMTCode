"""One-shot continuation of the submitted GT-head experiment on the local host.

Wait only for the already-authorized job chain; collect, audit, and build its UI.
This never submits training, changes a dataset, or selects a different model.
"""
from pathlib import Path, PurePosixPath
import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import tarfile
import time

HOST = 'zhanx0o@glogin.ibex.kaust.edu.sa'
REMOTE = '/ibex/user/zhanx0o/FMT_Task4C_GTHeadCoverage_20260917_r2/outputs/mainExp_Task4C_GTHeadCoverage_1.1'
OLD_REMOTE = '/ibex/user/zhanx0o/FMT_Task4C_GTHeadCoverage_20260917/outputs/mainExp_Task4C_GTHeadCoverage_1.1'
JOBS = '52000820,52000821,52000822,52000823,52000824,52000893,52000894,52000895,52000896,52000897'


def utc():
    return datetime.now(timezone.utc).isoformat()


def ssh(command):
    return subprocess.check_output(['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
        HOST, command], text=True, timeout=90)


def finish(workspace, input_root, wait_hours):
    workspace = Path(workspace).resolve()
    root = workspace/'outputs/mainExp_Task4C_GTHeadCoverage_1.1'
    root.mkdir(parents=True, exist_ok=True)
    status_file = root/'local_delivery_status.json'
    def status(stage, **extra):
        value = dict(stage=stage, at_utc=utc(), pid=os.getpid(), export_job='52000897', **extra)
        temporary = status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
        temporary.replace(status_file)
        print(json.dumps(value), flush=True)
    try:
        deadline = time.monotonic()+wait_hours*3600
        previous = None
        while True:
            state = ssh('sacct -j 52000897 --format=State,ExitCode -n -P').strip().splitlines()
            state = state[0].strip() if state else 'UNKNOWN|'
            if state != previous:
                status('waiting_for_submitted_export', scheduler=state)
                previous = state
            if state == 'COMPLETED|0:0':
                break
            if state.split('|')[0] in ('FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY'):
                raise RuntimeError('Export chain ended without a successful export: '+state)
            # A failed dependency can otherwise leave the export pending forever.
            chain = ssh('sacct -j 52000895,52000896 --format=JobIDRaw,State,ExitCode -n -P')
            for row in chain.splitlines():
                parts = row.split('|')
                if len(parts) >= 3 and '.' not in parts[0] and parts[1] in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'):
                    raise RuntimeError('A required GPU phase failed: '+row)
            if time.monotonic() >= deadline:
                raise TimeoutError('Submitted job still pending after the finite local continuation window')
            time.sleep(60)
        status('collecting_completed_predictions')
        members = ['scientific_config.json', 'data_audit.json', 'gpu_check.json', 'selection.lock.json', 'submission.json',
                   'runtime.jsonl', 'completion.json', 'viewer_package', 'final/c156/seed96721', 'logs']
        for flow in ('channel', 'tbl'):
            members += ['physical/'+flow+'/preparation.json', 'physical/'+flow+'/generation.json']
            members += ['physical/'+flow+'/'+role+'/metadata.npz' for role in ('train', 'validation', 'test')]
            members += ['physical/'+flow+'/added_'+role+'_index.npz' for role in ('train', 'test')]
        # All paths here are fixed experiment-owned files; no shell-built user text.
        ssh('cp /ibex/user/zhanx0o/FMT_Task4C_GTHeadCoverage_20260917_r2/config/mainExp_Task4C_GTHeadCoverage_1.1.json '+REMOTE+'/scientific_config.json')
        ssh('tar -C '+REMOTE+' -czf '+REMOTE+'/local_delivery.tar.gz '+' '.join(members))
        archive = root/'local_delivery.tar.gz'
        subprocess.run(['scp', '-o', 'BatchMode=yes', HOST+':'+REMOTE+'/local_delivery.tar.gz', str(archive)],
                       check=True, timeout=600)
        with tarfile.open(archive, 'r:gz') as stream:
            for member in stream.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe archive member: '+member.name)
            stream.extractall(root, filter='data')
        jobs = ssh('sacct -j '+JOBS+' --format=JobID,State,ExitCode,Elapsed,Start,End,Timelimit -n -P')
        (root/'scheduler_records.psv').write_text(jobs, encoding='utf8')
        (root/'first_attempt_runtime.jsonl').write_text(ssh('cat '+OLD_REMOTE+'/runtime.jsonl'), encoding='utf8')
        rows = {r.split('|')[0]: r.split('|') for r in jobs.splitlines() if r}
        for job in ('52000893_0', '52000893_1', '52000894', '52000895', '52000896', '52000897'):
            assert rows[job][1:3] == ['COMPLETED', '0:0'], rows[job]
        for job in ('52000820_0', '52000820_1'):
            assert rows[job][1:3] == ['FAILED', '1:0'], rows[job]
        for job in ('52000821', '52000822', '52000823', '52000824'):
            assert rows[job][1].startswith('CANCELLED'), rows[job]
        old_events = [json.loads(s) for s in (root/'first_attempt_runtime.jsonl').read_text().splitlines()]
        assert len(old_events) == 4
        status('independently_auditing_results')
        from experiments.Audit_Task4C_GTHeadResults_1_1 import audit
        assert json.loads((root/'scientific_config.json').read_text()) == json.loads(Path('config/mainExp_Task4C_GTHeadCoverage_1.1.json').read_text())
        audit(root, root/'scientific_config.json',
              workspace/'outputs/Verify_Task4C_GTHeadCoverage_1.1/catalog.json')
        status('building_complete_GT_viewer')
        from experiments.Visualize_Task4C_Bundles_3D import build_viewer
        from experiments.Build_Task4C_GTHeadViewer_1_1 import decorate
        from experiments.Build_FMT_AnalysisWorkbench_1_5 import build
        viewer = workspace/'outputs/Other_Task4C_GTHeadCoverage_1.1/viewer'
        package = root/'viewer_package'
        build_viewer(package, input_root, viewer, False)
        decorate(viewer, package, input_root)
        build(workspace/'outputs/Other_FMT_AnalysisWorkbench_1.4', viewer,
              workspace/'outputs/Other_FMT_AnalysisWorkbench_1.5', update_entry=True)
        verified = json.loads((root/'independent_results_audit.json').read_text())
        report = '# GT头部覆盖：自动构建完成\n\n'
        report += '全部预测及逐实例配额通过独立核验。132个GT各新增10测试＋30训练。\n\n'
        report += f"固定c156 / seed96721，新11320束测试F1：{verified['metrics']['test']['combined']['f1']:.9f}。这是单种子及修改后的数据集合。\n\n"
        report += '查看器已生成并更新入口；真实新数据的浏览器交互复核仍待执行，不能把自动构建称为浏览器测试通过。\n'
        (root/'delivery_report.md').write_text(report, encoding='utf8')
        status('built_and_audited_browser_review_pending',
               url='http://127.0.0.1:8767/Other_FMT_AnalysisWorkbench_1.5/index.html#results',
               test_f1=verified['metrics']['test']['combined']['f1'])
    except Exception as error:
        status('failed_old_page_retained', error=type(error).__name__+': '+str(error))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', required=True)
    p.add_argument('--input-root', required=True)
    p.add_argument('--wait-hours', type=float, default=24)
    a = p.parse_args()
    finish(a.workspace, a.input_root, a.wait_hours)
