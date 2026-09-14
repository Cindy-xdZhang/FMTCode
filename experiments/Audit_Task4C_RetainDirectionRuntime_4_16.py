"""Post-run Slurm/event reconciliation and model-free evidence packaging."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    root = Path('outputs/Verify_Task4C_RetainDirection_4.16')
    submissions = [json.loads(x) for x in (root/'submissions.jsonl').read_text().splitlines()]
    columns = ['JobID', 'JobIDRaw', 'State', 'ExitCode', 'NodeList', 'Start', 'End', 'Elapsed', 'AllocTRES']
    text = subprocess.check_output(['sacct', '-j', ','.join(r['job_id'] for r in submissions),
        '--noheader', '--parsable2', '--format='+','.join(columns)], text=True)
    rows = [dict(zip(columns, line.split('|'))) for line in text.splitlines() if line.strip()]
    expected = set()
    for r in submissions:
        expected.update([r['job_id']+'_'+str(i) for i in range(9)] if r['phase']=='train' else [r['job_id']])
    actual = {r['JobID']: r for r in rows if '.' not in r['JobID']}
    if set(actual) != expected:
        raise ValueError('Missing or unexpected scheduler processes')
    if len(actual) != 13:
        raise ValueError('Expected 13 successful processes')
    known_failure = None
    for key, r in actual.items():
        target = ('FAILED', '1:0') if key == known_failure else ('COMPLETED', '0:0')
        if (r['State'], r['ExitCode']) != target:
            raise ValueError('Unexpected scheduler status: '+str(r))
    events = [json.loads(x) for x in (root/'runtime_events.jsonl').read_text().splitlines()]
    durable = [json.loads(p.read_text()) for p in (root/'lifecycle_events').glob('*.json')]
    canonical = lambda seq: collections.Counter(json.dumps(r, sort_keys=True) for r in seq)
    if canonical(events) != canonical(durable) or len(events) != 26:
        raise ValueError('Aggregate events differ from durable per-event records')
    by_job = collections.defaultdict(list)
    for r in events: by_job[r['job']].append(r)
    for key, r in actual.items():
        event = by_job[r['JobIDRaw']]
        if len(event) != 2 or {e['state'] for e in event} != {'STARTED', 'ENDED'}:
            raise ValueError('Missing start/end event for '+key)
        ended = next(e for e in event if e['state']=='ENDED')
        if ended['exit_code'] != (1 if key == known_failure else 0):
            raise ValueError('Exit event differs from Slurm')
        if any(e['host'] != r['NodeList'] for e in event):
            raise ValueError('Runtime host differs from scheduler')
    independent = json.loads((root/'independent_audit.json').read_text())
    if independent['status'] != 'PASS' or len(independent['runs']) != 9:
        raise ValueError('Scientific prediction audit has not passed')
    expected_width = {'fmt_4_14': 233, 'fmt_v5_4_14': 398, 'fmt_objective_ntod_v2_4_14': 468}
    expected_params = {'fmt_4_14': 88514, 'fmt_v5_4_14': 109634, 'fmt_objective_ntod_v2_4_14': 118594}
    exported = []
    for path in sorted(root.glob('variants/*/runs/*/result.json')):
        r = json.loads(path.read_text()); encoder = r['encoder']
        if r['parameters'] != expected_params[encoder]:
            raise ValueError('Classifier capacity differs from preflight')
        norm = r['normalization']
        if encoder == 'conv3d':
            if norm is not None: raise ValueError('Conv3D normalization changed')
        elif len(norm['mean']) != expected_width[encoder] or len(norm['std']) != expected_width[encoder]:
            raise ValueError('Unexpected additional encoder channels')
        for split in ('training', 'validation', 'test'):
            for flow, metrics in [('pooled', r[split]['pooled']), *r[split]['per_flow'].items()]:
                exported.append(dict(encoder=encoder, seed=r['seed'], selected_epoch=r['selected_epoch'],
                    parameters=r['parameters'], split=split, flow=flow,
                    f1=metrics['f1'], average_precision=metrics['average_precision']))
    if len(exported) != 81:
        raise ValueError('Incomplete per-run metric table')
    with (root/'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0])); writer.writeheader(); writer.writerows(exported)
    report = dict(status='PASS', scheduler_processes=rows, successful_processes=13,
        preserved_failed_preflight=known_failure, runtime_events=len(events),
        all_events_match_per_event_files=True, parameters_and_feature_widths_verified=True,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'scheduler_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    archive = root/'metrics_evidence.tar.gz'
    included = []
    with tarfile.open(archive, 'w:gz') as package:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.is_symlink(): continue
            if path.suffix not in ('.json', '.jsonl', '.csv', '.out', '.err') and path.name != 'predictions.npz': continue
            package.add(path, arcname=str(path.relative_to(root))); included.append(str(path.relative_to(root)))
    manifest = dict(files=included, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        model_weights_included=False, source_commit=Path('SOURCE_COMMIT.txt').read_text().strip())
    (root/'evidence_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', processes=len(actual), events=len(events), files=len(included),
        archive_sha256=manifest['archive_sha256'])))


if __name__ == '__main__':
    main()
