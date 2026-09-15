"""Expand Slurm array accounting and match every executed process to both events."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expand(identifier):
    match=re.fullmatch(r'(\d+)_\[([^]]+)\]',identifier)
    if not match:return [identifier]
    out=[]
    for section in match[2].split('%')[0].split(','):
        ends=section.split('-');first=int(ends[0]);last=int(ends[-1])
        out.extend(match[1]+'_'+str(i) for i in range(first,last+1))
    return out


def audit(root,include_final=False):
    os.environ['TZ']='UTC'
    if hasattr(time,'tzset'):time.tzset()
    groups=[]
    for name in ('submission.json','cpu_prepare_submission.json','cpu_search_submission.json','cuda_verification_submission.json'):
        if (root/name).exists():
            values=read(root/name);values=values if isinstance(values,list) else [values]
            groups.extend((root,row) for row in values)
    if include_final:groups.extend((root/'final',r) for r in read(root/'final/submission.json'))
    columns=['JobIDRaw','JobID','JobName','State','ExitCode','NodeList','Start','End','Elapsed','Submit','AllocCPUS','ReqMem','ReqTRES','AllocTRES']
    raw=subprocess.check_output(['sacct','-n','-X','-P','-j',','.join(r['job_id'] for _,r in groups),
                                 '--format='+','.join(c+'%100' for c in columns)],text=True)
    filename='accounting_all_final.txt' if include_final else 'accounting_development_final.txt'
    (root/filename).write_text(raw)
    accounting={}
    for line in raw.splitlines():
        if not line.strip():continue
        row=dict(zip(columns,[v.strip() for v in line.split('|')]))
        for logical in expand(row['JobID']):
            if logical in accounting:raise ValueError('Duplicate logical accounting row')
            accounting[logical]=row
    records=[];event_count=0
    for folder,group in groups:
        count=group['count']
        for i in range(count):
            logical=group['job_id']+('_'+str(i) if count>1 else '')
            row=accounting.get(logical)
            if row is None:
                # A cancelled, never-materialized array may be reported only under its parent.
                row=accounting.get(group['job_id'])
                assert row and row['State'].startswith('CANCELLED') and count>1
            value={**row,'logical_job_id':logical,'phase':group['phase'],'expected_device':group['expected_device'],
                   'registered_commit':group['commit'],'registered_config_sha256':group['config_sha256']}
            if row['State'].startswith('CANCELLED'):
                assert not list((folder/'runtime').glob(row['JobIDRaw']+'_*_STARTED.json'))
                value['event_status']='Never started; cancellation accounting retained'
            else:
                assert row['State']=='COMPLETED' and row['ExitCode']=='0:0',(logical,row)
                array_index=str(i) if count>1 else 'None'
                events=[]
                for state in ('STARTED','ENDED'):
                    path=folder/'runtime'/f'{row["JobIDRaw"]}_{array_index}_{group["phase"]}_{state}.json'
                    event=read(path)
                    assert event['job_id']==row['JobIDRaw'] and event['state']==state
                    assert event['hostname']==row['NodeList']
                    assert event['commit']==group['commit']
                    assert event['config_sha256']==group['config_sha256']
                    if group['expected_device']=='CPU':assert event['device']=='CPU'
                    elif group['expected_device']=='GPU':assert event['device']!='CPU'
                    else:assert group['expected_device'].lower() in event['device'].lower()
                    instant=datetime.fromisoformat(event['timestamp'].replace('Z','+00:00'))
                    start=datetime.fromisoformat(row['Start']).replace(tzinfo=timezone.utc)
                    end=datetime.fromisoformat(row['End']).replace(tzinfo=timezone.utc)
                    assert start.timestamp()-2<=instant.timestamp()<=end.timestamp()+2
                    if state=='ENDED':assert event['exit_code']==0
                    events.append({'file':str(path.relative_to(root)),'sha256':sha(path),'timestamp':event['timestamp']})
                assert events[0]['timestamp']<=events[1]['timestamp']
                value['events']=events;value['event_status']='PASS';event_count+=2
            records.append(value)
    result={'status':'PASS','include_final':include_final,'registered_processes':len(records),
            'completed_processes':sum(r['State']=='COMPLETED' for r in records),
            'cancelled_before_start':sum(r['State'].startswith('CANCELLED') for r in records),
            'runtime_events_verified':event_count,'accounting_timezone':'UTC','accounting_sha256':sha(root/filename),
            'auditor_source_sha256':sha(__file__),'processes':records}
    name='runtime_audit_all.json' if include_final else 'runtime_audit_development.json'
    (root/name).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='processes'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--include-final',action='store_true');args=parser.parse_args()
    audit(args.root,args.include_final)
