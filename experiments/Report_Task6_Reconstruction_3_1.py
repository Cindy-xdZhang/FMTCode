"""Check and summarize all recovery runs, including the unchanged historical test."""
import argparse
import csv
import datetime
import json
from pathlib import Path

import numpy as np
from FMT_Utils.FlowMapData_3D import sha256,write_json


def summarize(root):
    config=root/"config.frozen.json"
    spec=json.loads(config.read_text())
    audit=json.loads((root/"final_audit.json").read_text())
    assert audit["passed"] and audit["rows"]==2592 and audit["runs"]==54
    assert audit["provenance"]["config_sha256"]==sha256(config)
    with (root/"metrics.csv").open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    index={(r["dataset"],r["arm"],int(r["seed"]),r["role"],int(r["scale_id"])):r for r in rows}
    assert len(index)==2592
    summaries=[]; run_rows=[]
    for dataset in spec["datasets"]:
        data_audit=json.loads((root/"data"/dataset/"audit.json").read_text())
        assert data_audit["passed"] and data_audit["counts"]==spec["sampling"]["retained_samples"]
        summary={"dataset":dataset}
        for arm in spec["arms"]:
            for seed in spec["seeds"]:
                result=json.loads((root/"runs"/dataset/arm/str(seed)/"result.json").read_text())
                assert result["provenance"]["config_sha256"]==sha256(config)
                assert result["provenance"]["git_commit"]==audit["provenance"]["git_commit"]
                assert result["training"]["updates"]==15008
                assert result["training"]["train_examples_exposed"]==7680000
                assert 0<result["training"]["selected_step"]<=15008
                assert result["training"]["selected_validation_rmse_r"]<.8
                fitting=json.loads((root/"runs"/dataset/arm/str(seed)/"fit_check/result.json").read_text())
                assert fitting['updates']==8000 and fitting['metric']['samples']==512
                for metric in result["metrics"]:
                    flat=index[(dataset,arm,seed,metric["role"],metric["scale_id"])]
                    for k,v in metric.items():
                        if k not in ("role","scale_id") and not isinstance(v,list):
                            np.testing.assert_allclose(float(flat[k]),v,rtol=1e-12,atol=1e-12)
                run_rows.append(dict(dataset=dataset,arm=arm,seed=seed,device=result["device"],
                    actual_job_id=result["provenance"]["job_id"],
                    validation_rmse_r=result["training"]["selected_validation_rmse_r"],
                    selected_step=result["training"]["selected_step"],
                    trainable_parameters=result['training']['initialization']['trainable_parameters'],
                    initialization_validation_rmse_r=result["training"]["curve"][0]["validation_rmse_r"],
                    fit_rmse_r=result["fit_check"]["position_rmse_r"]))
            for role in ("test","unseen_scale","legacy_test"):
                values=np.array([float(index[(dataset,arm,s,role,-1)]["position_rmse_r"]) for s in spec["seeds"]])
                summary[f"{arm}_{role}_mean"]=float(values.mean())
                summary[f"{arm}_{role}_std"]=float(values.std(ddof=1))
                summary[f"{arm}_{role}_max_seed"]=float(values.max())
        summaries.append(summary)
    macro={k:float(np.mean([r[k] for r in summaries])) for k in summaries[0] if k.endswith("_mean")}
    goal={role:all(float(r["position_rmse_r"])<1 for r in rows if r["arm"]=="signed_fmt_vae" and r["role"]==role and int(r["scale_id"])==-1) for role in ("test","unseen_scale","legacy_test")}
    report=dict(experiment=spec["experiment"],audit_passed=True,config_sha256=sha256(config),
        git_commit=audit["provenance"]["git_commit"],metrics_sha256=sha256(root/"metrics.csv"),
        per_flow=summaries,macro=macro,runs=run_rows,all_signed_fmt_runs_below_one=goal,
        metric="Unchanged corresponding 7 x 31 noninitial Euclidean position RMSE / initial radius",
        aggregation="Within-flow three-seed mean, then equal-weight nine flows",
        std="Sample standard deviation across three optimizer seeds, not bootstrap confidence intervals",
        input_contract="signed_fmt10:399 signed real/imaginary coefficients; latent dimension192; decoder only receives latent",
        comparison_scope="New dataset and training; legacy_test fixes the old evaluation trajectories but is an already used benchmark")
    write_json(root/"summary.json",report)
    with (root/"per_flow_summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(json.dumps(dict(goals=goal,macro=macro,flows=len(summaries),runs=len(run_rows)),indent=2))
    return report


def record_log(report,root):
    """Append method-level conclusions to the project's sole experiment log."""
    marker='mainExp_Task6_Reconstruction_3.1：54组训练与最终审计完成'
    path=Path('docs/experiment_log.md')
    if marker in path.read_text(encoding='utf-8'):
        raise FileExistsError('The final log already exists; preserve it instead of appending a duplicate')
    old=json.loads(Path('outputs/mainExp_Task6_PrimitiveVAE_2.1/summary.json').read_text())
    prior={r['dataset']:r for r in old['per_flow']}
    def cell(row,prefix):
        return f"{row[prefix+'_mean']:.6g} ± {row[prefix+'_std']:.3g}"
    lines=[f"\n\n### {datetime.date.today().isoformat()} — {marker}\n",
        f"证据：训练代码`{report['git_commit']}`，配置`config/mainExp_Task6_Reconstruction_3.1.json`，实际配置SHA256 `{report['config_sha256']}`。54/54训练完成，最终从预测独立复算2592条指标；本地再次核对CSV、逐次JSON、训练更新/曝光、正步数模型选择及validation门槛。\n",
        "主指标保持原定义：七条对应路径线31个非初始时刻的三维位置均方根误差，除以初始邻居半径r。下表为三个优化种子的均值±样本标准差；时间、点身份及坐标不做目标辅助变换。目标按每个流场/种子的全测试集RMSE判断，不是每个点或每个primitive的最大误差。\n"]
    for role,label in [('test','新test'),('legacy_test','原固定legacy_test'),('unseen_scale','未见尺度test')]:
        maximum=max(r[f'signed_fmt_vae_{role}_max_seed'] for r in report['per_flow'])
        lines.append(f"- {label}：27个signed-FMT模型是否全部RMSE/r<1：**{report['all_signed_fmt_runs_below_one'][role]}**；最大单次模型指标{maximum:.9g}。")
    lines+=['\n同一历史测试轨迹的比较（该benchmark已使用过）：\n',
        '| 流场 | 原FMT-VAE 2.1 | 新signed-FMT-VAE 3.1 | 同版Raw-VAE 3.1 |',
        '|---|---:|---:|---:|']
    for row in report['per_flow']:
        lines.append(f"| {row['dataset']} | {cell(prior[row['dataset']],'fmt_all_vae_test')} | {cell(row,'signed_fmt_vae_legacy_test')} | {cell(row,'raw_vae_legacy_test')} |")
    lines+=['\n新采样的测试primitive及未见尺度：\n',
        '| 流场 | signed-FMT test | Raw test | signed-FMT未见尺度 | Raw未见尺度 |',
        '|---|---:|---:|---:|---:|']
    for row in report['per_flow']:
        lines.append('| '+row['dataset']+' | '+' | '.join(cell(row,k) for k in ['signed_fmt_vae_test','raw_vae_test','signed_fmt_vae_unseen_scale','raw_vae_unseen_scale'])+' |')
    lines+=['\n初始化和神经网络训练分别列出（signed-FMT三个种子的均值）：\n',
        '| 流场 | 初始化validation | 正训练步选中模型validation | 512样本拟合误差 |',
        '|---|---:|---:|---:|']
    for dataset in prior:
        runs=[r for r in report['runs'] if r['dataset']==dataset and r['arm']=='signed_fmt_vae']
        lines.append('| '+dataset+' | '+' | '.join(f"{np.mean([r[k] for r in runs]):.9g}" for k in ['initialization_validation_rmse_r','validation_rmse_r','fit_rmse_r'])+' |')
    raw_lower=sum(r['raw_vae_test_mean']<r['signed_fmt_vae_test_mean'] for r in report['per_flow'])
    lines+=[f"\n本轮新test按流场三种子均值比较，Raw-VAE误差低于signed-FMT-VAE的流场数为{raw_lower}/9。九流场等权宏平均：signed-FMT {report['macro']['signed_fmt_vae_test_mean']:.9g}，Raw {report['macro']['raw_vae_test_mean']:.9g}。这次回答完整几何能否被重建；该结果中的Raw对照必须共同保留。\n",
        "修订说明：旧2.1使用原161维Gram/模长与邻居排序特征，且VAE从随机初始化开始；本版使用保留方向、相位、七线身份的399维signed_fmt10，192维潜变量及只在train拟合的线性几何初始化，再训练编码/解码残差网络和后验方差，每流场train扩大至240000。新成绩属于新编码，原fmt_all源码及2.1历史结果保持。多项设计同时改变，单项贡献需要独立消融；初始化已达到的精度单独报告。\n",
        "所有几何输出经过192维潜变量，解码器只接收潜变量。每组512个train primitive拟合8000更新，然后全量训练重新初始化，32epoch=15008更新、768万训练样本曝光；beta=1e-5全程为正，validation选择step>0且误差<0.8后才读test。仅保留指标/预测，未写出模型checkpoint。\n",
        "第一性原理排查的可核验证据见Verify_Task6_ReconstructionAudit_2.2：独立积分和缓存回放只检查每流场4个固定train样本，未发现能解释13量级误差的积分错误；原分类编码不能普遍唯一恢复带符号坐标的反例及完整复系数回环已通过测试。原问题是重建输入信息与训练设计需要修复，不能把分类表征的成绩直接当成坐标重建能力。\n",
        "新test使用新primitive采样种子，源时间块沿用原协议；legacy_test为同一已用benchmark。训练、模型选择与评价角色保持区分。额外按真实几何总转角选择的弯曲图只作可视化核对，原固定样例和全量指标均保留。\n",
        f"完整证据：`{root.as_posix()}/{{summary.json,per_flow_summary.csv,metrics.csv,final_audit.json,config.frozen.json,slurm_status_final.txt}}`及`runs/`逐次结果；复算代码`experiments/Report_Task6_Reconstruction_3_1.py`，图形规则`docs/Task6_reconstruction_artifacts_3.1.md`。\n"]
    with path.open('a',encoding='utf-8') as f:f.write('\n'.join(lines))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('outputs/mainExp_Task6_Reconstruction_3.1'))
    parser.add_argument('--record-log',action='store_true')
    args=parser.parse_args();report=summarize(args.root)
    if args.record_log: record_log(report,args.root)
