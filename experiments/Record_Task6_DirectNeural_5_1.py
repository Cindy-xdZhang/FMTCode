"""Post-run records only: no fitting, prediction, model selection, or NPZ reads."""
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_once(path, marker, body):
    path = Path(path)
    if marker not in path.read_text(encoding='utf-8'):
        with path.open('a', encoding='utf-8') as f:
            f.write('\n'+marker+'\n\n'+body+'\n')


def main():
    config = Path('config/Verify_Task6_DirectNeural_5.1.json')
    spec = read(config)
    root = Path(spec['output_root'])
    assert digest(config) == digest(root/'config.frozen.json')
    jobs = [json.loads(s) for s in (root/'submissions.jsonl').read_text().splitlines()]
    # This recorder itself is separately registered after submission.
    extra = root/'recorder_submission.json'
    all_jobs = jobs+([read(extra)] if extra.exists() else [])
    ids = ','.join(j['job_id'] for j in all_jobs)
    raw = subprocess.check_output(['sacct', '-j', ids, '-n', '-P', '-X',
        '--format=JobID,JobIDRaw,JobName,State,Submit,Start,End,NodeList,ExitCode'], text=True)
    events = {p.name: read(p) for p in (root/'events').glob('*.json')}
    lines = ['科学commit `'+jobs[0]['git_commit']+'`；配置SHA256 `'+digest(config)+'`。', '',
        '| Job / raw ID | 阶段 | 提交 | 开始 | 结束 | 节点 / GPU | 状态/退出码 |',
        '|---|---|---|---|---|---|---|']
    for line in raw.splitlines():
        f = line.split('|')
        if len(f) < 9:
            continue
        job, raw_id, name, state, submit, start, end, node, code = f[:9]
        phase = name.removeprefix('t6-direct-')
        index = job.split('_')[-1] if '_' in job else '0'
        event = events.get(f'{phase}_{index}.json', {})
        device = event.get('gpu', 'CPU')
        lines.append(f'| {job} / {raw_id} | {phase} | {submit} | {start} | {end} | {node} / {device} | {state}/{code} |')
    audit_path = root/'final_audit.json'
    audit = read(audit_path) if audit_path.exists() else dict(passed=False, reason='Final audit absent; inspect failed/cancelled prerequisites')
    lines += ['', '最终审计状态：'+json.dumps(audit, ensure_ascii=False)]
    append_once('docs/ibex_run_registry.md', '### Task6 5.1 全部实验作业终态自动登记', '\n'.join(lines))
    report = dict(time=datetime.datetime.now().astimezone().isoformat(),
        scientific_commit=jobs[0]['git_commit'], config_sha256=digest(config), audit=audit, sacct=raw)
    if not audit.get('passed'):
        append_once('docs/experiment_log.md', '### Task6 5.1 自动结束记录：结果不完整',
            '实验存在失败或独立审计未完成，不输出删除失败样本后的宏平均。详见ibex_run_registry及outputs/Verify_Task6_DirectNeural_5.1/final_audit.json。\n'+json.dumps(audit, ensure_ascii=False))
    else:
        assert audit['selection_sha256'] == digest(root/'selection.json') == digest(root/'selection.before_test.json')
        with (root/'metrics.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == audit['rows']
        comparisons = ['fmt', 'raw_matched', 'raw_selected', 'pca']
        summary = []
        for n in spec['train_sizes']:
            for role in ('test', 'unseen_scale'):
                per_flow = {}
                for method in comparisons:
                    values = [r for r in rows if int(r['train_size']) == n and r['role'] == role
                        and r['comparison'] == method and int(r['scale_id']) == -1]
                    assert len(values) == len(spec['datasets'])*len(spec['seeds'])
                    per_flow[method] = {}
                    for d in spec['datasets']:
                        group = [r for r in values if r['dataset'] == d]
                        assert sorted(int(r['seed']) for r in group) == sorted(spec['seeds'])
                        per_flow[method][d] = math.fsum(float(r['position_rmse_r']) for r in group)/len(group)
                item = dict(train_size=n, split=role)
                item.update({m: math.fsum(per_flow[m].values())/len(spec['datasets']) for m in comparisons})
                item['fmt_wins_vs_raw_matched'] = sum(per_flow['fmt'][d] < per_flow['raw_matched'][d] for d in spec['datasets'])
                item['relative_gain_vs_raw_matched_percent'] = 100*(item['raw_matched']-item['fmt'])/item['raw_matched']
                summary.append(item)
        report['macro'] = summary
        with (root/'macro.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        text = [f"科学commit `{jobs[0]['git_commit']}`，config SHA256 `{digest(config)}`，selection SHA256 `{audit['selection_sha256']}`。",
            f"独立复算通过：{audit['models']}个最终模型（含独立PCA），{audit['rows']}行记录。", '',
            '| 训练样本 | split | FMT | 同配置Raw | 独立选Raw | PCA192 | FMT相对同配置Raw增益 | 胜出流场数 |',
            '|---:|---|---:|---:|---:|---:|---:|---:|']
        for r in summary:
            text.append(f"| {r['train_size']} | {r['split']} | {r['fmt']:.8f} | {r['raw_matched']:.8f} | {r['raw_selected']:.8f} | {r['pca']:.8f} | {r['relative_gain_vs_raw_matched_percent']:.3f}% | {r['fmt_wins_vs_raw_matched']}/{len(spec['datasets'])} |")
        text += ['', '指标为RMSE/r，越低越好。先每流场三种子平均，再九流场等权平均；正增益表示FMT误差更低，负增益表示更高。',
            '这是已用benchmark；完整16频编码仍可逆，192维潜变量由网络学习。结果不等同于旧161维fmt_all，也不说明客观性。独立PCA从未参与神经网络。原3.1/4.1结果不变。']
        append_once('docs/experiment_log.md', '### Task6 5.1 最终测试与独立审计自动记录', '\n'.join(text))
    (root/'automatic_end_record.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
