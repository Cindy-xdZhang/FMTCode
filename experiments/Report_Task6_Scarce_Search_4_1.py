"""Independently check all validation candidates and record the frozen selection."""
import argparse
import json
from pathlib import Path

import numpy as np

from FMT_Utils.FlowMapData_3D import sha256,write_json


def main(root):
    spec=json.loads((root/'config.frozen.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    assert sha256(root/'selection.json')==sha256(root/'selection.before_test.json')
    rows=selection['rows'];assert len(rows)==324 and not selection['test_read']
    scores=[]
    for n in spec['train_sizes']:
        for c in ['r0','r1','r2','r3','c0','c1','c2','c3','c4','c5','c6','c7']:
            subset=[r for r in rows if r['train_size']==n and r['candidate']==c]
            assert len(subset)==9 and len({r['dataset'] for r in subset})==9
            value=float(np.mean([r['validation_rmse_r'] for r in subset]))
            scores.append(dict(train_size=n,candidate=c,validation_rmse_r=value))
            if n==spec['primary_train_size']:
                arm='raw_vae' if c.startswith('r') else 'signed_fmt_vae'
                np.testing.assert_allclose(selection['scores'][arm+'_'+c],value,rtol=1e-14)
    for arm,candidates in [('fmt',spec['candidates']),('raw_selected',[dict(id='r'+str(i)) for i in range(4)])]:
        assert selection[arm]['id']==min(candidates,key=lambda c:(next(s['validation_rmse_r'] for s in scores
            if s['candidate']==c['id'] and s['train_size']==spec['primary_train_size']),c['id']))['id']
    write_json(root/'search_summary.json',dict(scores=scores,selection_sha256=sha256(root/'selection.json'),test_read=False))
    log=Path('docs/experiment_log.md');marker='Verify_Task6_ScarceGeneralization_4.1 完整验证选择'
    assert marker not in log.read_text(encoding='utf-8')
    lines=['\n\n### 2026-09-11 — '+marker+'\n',
        f"324个候选全部完成。代码`{selection['provenance']['git_commit']}`；选择文件SHA256 `{sha256(root/'selection.json')}`在读取test前固定。搜索种子94110，每个初始化和网络只使用指定训练子集。\n",
        '| 候选 | 256样本validation | 1024样本validation（主设置） | 4096样本validation |',
        '|---|---:|---:|---:|']
    for c in ['r0','r1','r2','r3','c0','c1','c2','c3','c4','c5','c6','c7']:
        lines.append('| '+c+' | '+' | '.join(f"{next(s['validation_rmse_r'] for s in scores if s['candidate']==c and s['train_size']==n):.9g}" for n in spec['train_sizes'])+' |')
    lines += ['\n全局选择c6：16频、dropout=0.25、权重衰减0.01、不加输入噪声。独立选择的Raw为r2，与FMT正则化相同；两个Raw角色共享模型。',
        '\n1024样本时，所选FMT的validation误差比原Raw低15.777%，比同正则化/独立选优Raw低0.378489%。但256/4096样本时，同一个FMT候选的validation分别比r2高1.66679%/0.124111%。这些是搜索种子的validation结果，不能充当测试证据。',
        '\n10频候选总体仍落后Raw；增加噪声也没有在预注册主设置获胜。最终阶段固定c6/r0/r2，以三个新训练子集及随机种子重新拟合，独立测试。']
    with log.open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(selection['scores'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/Verify_Task6_ScarceGeneralization_4.1'))
    args=p.parse_args();main(args.root)
