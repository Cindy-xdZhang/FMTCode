"""Append actual Slurm completion and device records for the frozen experiment."""
import argparse
import csv
import datetime
import json
from pathlib import Path


def record(root,stage):
    ledger=[json.loads(line) for line in (root/f'submissions_{stage}.jsonl').read_text().splitlines()]
    with (root/f'slurm_status_{stage}.txt').open(newline='') as f:
        statuses=list(csv.DictReader(f,delimiter='|'))
    log=Path('docs/ibex_run_registry.md')
    marker=f'Task6小训练集4.1 {stage}完整结束记录'
    assert marker not in log.read_text(encoding='utf-8')
    lines=[f'\n\n### {datetime.date.today().isoformat()} — {marker}\n',
        f"实验Verify_Task6_ScarceGeneralization_4.1 / Task6；配置`{ledger[0]['config']}`、SHA256 `{ledger[0]['config_sha256']}`、代码`{ledger[0]['git_commit']}`。以下时间均为+03:00；原提交、调度修改和开始记录保留。\n",
        '| Job / 实际进程ID | 阶段/流场/训练数 | 提交 | 开始 | 结束 | 节点/设备 | 最终状态 |',
        '|---|---|---|---|---|---|---|']
    for entry in ledger:
        phase=entry['phase'];job=entry['job_id']
        entries=[r for r in statuses if r['JobID']==job or r['JobID'].startswith(job+'_')]
        assert entries
        metadata={}
        if phase in ('search','final'):
            for path in (root/phase).glob('*/*/started.json'):
                item=json.loads(path.read_text())
                metadata[item['provenance']['array_id']]=item
        for status in entries:
            assert status['State']=='COMPLETED' and status['ExitCode']=='0:0',status
            label=status['JobID'];description=phase;device='CPU'
            if metadata:
                item=metadata[label.split('_')[-1]]
                label+=' / '+item['provenance']['job_id']
                description+=f" / {item['dataset']} / {item['train_size']}"
                device=item['device']
            lines.append('| '+' | '.join([label,description,status['Submit'],status['Start'],status['End'],
                status['NodeList']+' / '+device,status['State']+'/'+status['ExitCode']])+' |')
    if stage=='search':
        selection=json.loads((root/'selection.json').read_text())
        assert len(selection['rows'])==324 and selection['test_read'] is False
        lines.append('\n324个模型的训练/验证搜索完成，选择记录已写入selection.json；方法级结论见experiment_log.md，未用test参与选择。')
    else:
        audit=json.loads((root/'final_audit.json').read_text())
        assert audit['passed'] and audit['rows']==9396
        lines.append(f"\n实际唯一最终模型{audit['models']}个，独立复算{audit['rows']}条比较/尺度记录通过。selection SHA256 `{audit['selection_sha256']}`；未产生checkpoint。前台只读监测已随审计结束，无定时任务。")
    with log.open('a',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    print(marker)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['search','final'])
    p.add_argument('--root',type=Path,default=Path('outputs/Verify_Task6_ScarceGeneralization_4.1'))
    args=p.parse_args();record(args.root,args.stage)
