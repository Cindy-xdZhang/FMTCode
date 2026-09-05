# Task3-3D 主实验：FMT 提高监督 IVD 涡区域识别

本文合并原先分散的 Task3-3D 主实验文档。逐实验、逐作业的完整历史分别保留在
[`experiment_log.md`](experiment_log.md) 和
[`ibex_run_registry.md`](ibex_run_registry.md)；本文只维护 Task3 主方法的演进、
已确认历史结果、当前统一配置主结果和证据边界。

## 已确认结论与当前主表状态

Task3 是 whole-field IVD p95 标签监督的涡/非涡二分类，不是
streamwise/spanwise/hairpin 多分类。当前论文主结果是
`mainExp_Task3_3D_9.2_uniform_confirmation`：全部10个数据条目使用同一个
`aivd1w3_dft` FMT block和同一个64D auxiliary residual。相对同结构、同输入维度的
Raw-PCA residual，dataset-macro F1 从`.63984`提高到`.85834`，增益`+.21849`；
Average Precision从`.68910`提高到`.92357`，增益`+.23447`。F1/AP在10/10数据
条目为正，F1在7/7 physical family和5/5 paired training seeds为正。

历史逐family结果`mainExp_Task3_3D_8.1`为F1`.69131→.87136`、增益`+.18005`，
AP`.74392→.94383`、增益`+.19991`。它的绝对FMT F1略高，但使用不同的
confirmation population与Raw对照，且每个family配方不同，因此只保留为补充证据。

旧逐family冻结方法在另一套未见空间 primitive population（7.2）得到 F1/AP 增益
`+.17522/+.20146`。7.2 与 8.1 的平均 F1/AP 增益为 `+.17764/+.20068`。因此，现有
证据支持“FMT 在当前 10 个 3D 数据条目的监督 IVD-p95 二分类中稳定提高性能”，但
不构成对任意 3D 流场的数学保证。

## 固定协议

- 数据：Channel observer、Half-cylinder Re160/Re640/Re6400、Tangaroa、
  Delta-wing resampled、Delta-wing original LBM、F-22、Boeing 747、Smoke
  buoyancy，共 10 个数据条目、7 个 physical family。
- 标签：`IVD(seed) >= percentile95(whole-field IVD volume)`；标签定义与 Task1、
  Task2、Task5 一致。
- 历史 8.1 比较：每个 physical family 在 development 上冻结一套 FMT feature
  与训练配方；有效 FMT 维度为 1--171D，并以完全相同的维度、residual-head 结构
  和可训练参数量构造 train-only Raw-PCA residual。Raw 主干冻结。统一 268D
  `all_plus_gram_kinematic` 是早期 3.2/Task5 配方，不是 8.1 的有效模型组合。
- 当前 9.2 比较：所有数据条目固定`aivd1w3_dft`（1D）与64D auxiliary
  residual；Raw-PCA与FMT两臂使用同结构、同容量网络。Pathline统一为7条线、RK4、
  `offset_grid_scale=.5`、`dt_scale=.25`、积分48步、每线重采样32点。
- 选择：feature、训练配方、checkpoint、threshold、residual scale、Raw
  normalization 和 Raw-PCA transform 只在 development 数据上确定。
- 确认：confirmation 只做冻结推理，不训练、不选择。7.2 和 8.1 只改变此前未见的
  空间 seed-grid phase。
- 指标：F1 与 Average Precision（AP）；同时报告 dataset macro、family macro、
  每个数据条目和 paired seed 的增益。

## 主实验演进

| 版本 | 标签/方法变化 | 主要结果 | 结论与状态 |
|---|---|---|---|
| `mainExp_Task3Universality_1.1` | local-IVD 标签；由 joint Raw+FMT 改为冻结 Raw 的 additive FMT residual | 8 个条目、5 个 family，24/24 dataset-seed 的 F1/AP 高于 Raw/Raw-wide；F-22 最小 F1 增益仅 `+.00159` | 证明 residual 结构可避免 joint 模型在 F-22 的失败；标签和对照仍不够严格，已被后续版本取代 |
| `mainExp_Task3Universality_2.2` | 268D FMT；constrained-AP validation selection | 8/8 条目 F1/AP method-mean 增益均 `>=.02` | 正方向成立，但选择规则显式要求 FMT 增益，存在审稿风险，不作为当前主表 |
| `mainExp_Task3NewFlows_2.3` | 增加 Boeing 747 与 Smoke buoyancy | Boeing F1/AP `+.0651/+.1617`；Smoke `+.0299/+.0433` | 两个新增 family 均支持正增益；仍沿用旧标签和选择规则 |
| `mainExp_Task3_3D_3.1` | 所有方法只按自身 validation AP 选 epoch；增加同结构 Raw-PCA residual | 相对原始 Raw 的 F1/AP 均值增益 `+.0701/+.1176`；相对 Raw-PCA 为 `-.0009/+.0117` | 修订旧结论：FMT 提高原始 Raw，但在该 local-IVD 标签下没有超过强 Raw-only 对照 |
| `mainExp_Task3_3D_3.2_global_ivd` | 唯一科学变量改为统一 whole-field IVD p95 | 相对 Raw-PCA 的 dataset-macro F1/AP 增益 `+.0502/+.0445`；F1 9/10、AP 8/10 为正；F-22 为负 | 证明结论依赖标签协议；p95 下宏平均明确提高，但逐流场还不一致 |
| `mainExp_Task3_3D_4.1` | 冻结 p95，FMT residual family search | F1/AP 增益 `+.1000/+.1190`，10/10 条目为正 | 中间方法基线；保留用于说明方法演进 |
| `mainExp_Task3_3D_5.1` | anchored FMT，新空间 population | F1/AP `.71683→.85274/.77720→.92692`，增益 `+.13591/+.14971`；10/10 条目为正 | 改善 4.1，但未达到预注册 F1 `+.15` 目标，按未达目标保留 |
| `mainExp_Task3_3D_5.2` | 第三空间 population 预注册 | 无完整结果 | 只保留预注册事实，不产生性能结论 |
| `mainExp_Task3_3D_6.1` | 冻结 22.1 anchored feature；第四空间 population | F1/AP `.69480→.86546/.75263→.94465`，增益 `+.17066/+.19202`；10/10、7/7 为正 | 首次在独立确认中超过 F1 `+.15` 目标；历史强证据 |
| `mainExp_Task3_3D_7.1` | 最终调参栈；第五空间 population | 无性能结果 | 作业链后来终止，不得从该版本推导结论 |
| `mainExp_Task3_3D_7.2` | adaptive portfolio；第六空间 population | F1/AP `.68584→.86106/.73624→.93770`，增益 `+.17522/+.20146` | 通过 `+.15`，未达 `+.20` 扩展目标；10/10、7/7、2/2 为正 |
| `mainExp_Task3_3D_8.1` | 与 7.2 同一冻结模型组合；第七空间 population | F1/AP `.69131→.87136/.74392→.94383`，增益 `+.18005/+.19991` | 历史逐family强证据；通过`+.15`，未达`+.20`扩展目标；只作补充结果 |
| `Verify_Task3_UniformFMT_9.1` | 从9套既有FMT配方中选择一套跨全部数据不变的配方；只用development | 冻结`aivd1w3_dft`；development F1/AP增益`+.19874/+.21734`，10/10条目为正 | development选择及独立审计`PASS`；不作为confirmation性能 |
| `mainExp_Task3_3D_9.2` | 单一`aivd1w3_dft`；新confirmation ordinals 8–9；5 paired seeds | F1/AP `.63984→.85834/.68910→.92357`，增益`+.21849/+.23447` | **当前论文统一配置主结果**；100条记录独立审计`PASS`，10/10、7/7、5/5 F1正 |

## 历史补充表：8.1逐family配方

下表完整保留用于补充材料和可视化。它不能与已完成的9.2统一配置结果混合计算，
也不得继续标作当前论文主表。

| Flow | Raw-PCA F1 | Raw+FMT F1 | F1增益 | Raw-PCA AP | Raw+FMT AP | AP增益 |
|---|---:|---:|---:|---:|---:|---:|
| Channel observer | .1182±.0065 | .8428±.0043 | **+.7246±.0109** | .0696±.0066 | .8847±.0095 | **+.8151±.0029** |
| Half-cylinder Re160 | .7090±.0207 | .8672±.0118 | **+.1582±.0325** | .7827±.0007 | .9532±.0073 | **+.1704±.0066** |
| Half-cylinder Re640 | .6739±.0005 | .8775±.0030 | **+.2036±.0035** | .7315±.0062 | .9490±.0054 | **+.2175±.0008** |
| Half-cylinder Re6400 | .5365±.0122 | .7768±.0004 | **+.2403±.0126** | .5649±.0056 | .8787±.0008 | **+.3138±.0064** |
| Tangaroa | .7494±.0102 | .8520±.0022 | **+.1026±.0079** | .7819±.0182 | .9305±.0061 | **+.1487±.0121** |
| Delta-wing resampled | .8865±.0006 | .9269±.0005 | **+.0404±.0011** | .9616±.0005 | .9832±.0002 | **+.0216±.0006** |
| Delta-wing original LBM | .8668±.0027 | .9031±.0021 | **+.0363±.0048** | .9491±.0010 | .9769±.0020 | **+.0278±.0010** |
| F-22 | .7997±.0077 | .9218±.0019 | **+.1221±.0059** | .8812±.0030 | .9824±.0005 | **+.1011±.0025** |
| Boeing 747 | .7623±.0029 | .8778±.0011 | **+.1155±.0040** | .8240±.0172 | .9440±.0004 | **+.1200±.0168** |
| Smoke buoyancy | .8107±.0032 | .8676±.0028 | **+.0569±.0060** | .8926±.0035 | .9558±.0014 | **+.0631±.0049** |
| **Dataset macro** | **.6913** | **.8714** | **+.1800** | **.7439** | **.9438** | **+.1999** |
| **Family-macro gain** | — | — | **+.1944** | — | — | **+.2152** |

## 审计证据

| 版本 | 完整性与独立审计 |
|---|---|
| 6.1 | 40/40 唯一冻结推理记录；summary/per-run SHA-256 `f46a307a…fb756` / `bc5a61b9…ef591`；独立重算误差 `<1e-12` |
| 7.2 | 完整作业链全部 exit 0；summary/per-run/audit SHA-256 `e89751f9…5248` / `7c1ecc37…3901` / `cdadc4d6…6350`；独立重算最大差 `2.78e-17` |
| 8.1 | 初次 evaluate 在推理前因依赖包缺 Raw 文件失败，产生 0 条指标；使用原 checkpoint 的 SHA-256 精确复制建立依赖闭包后重跑，不重训、不改模型。summary/per-run/audit SHA-256 `0066299c…07bd` / `78d19ec3…bc52` / `eabc384a…edb3`；独立重算最大差 `1.11e-16` |
| 9.2 | 100/100逐次记录；summary/per-run/audit SHA-256 `f577a632…6ab9` / `6e80adfa…50a9` / `e8af36e4…7a8e`；独立审计`PASS`。审计前100个临时checkpoint，cleanup精确删除100并确认剩余0 |

`Verify_Task23PaperTableConsistency_1.1` 还对 Task2/Task3 共 520 条核心记录重新配对
汇总，正式 summary 最大差 `2.22e-16`，论文表 135 个显示值全部通过四位小数检查。
详见 [`Audit_experiments.md`](Audit_experiments.md)。

## 可以与不能写入论文的结论

可以作为论文主结论写：在固定 whole-field IVD p95、同结构同容量 Raw-PCA residual
对照和冻结空间confirmation下，一套跨全部10个数据条目不变的FMT配方将F1提高
`+.2185`、AP提高`+.2345`，且所有数据条目方向为正。逐family 7.2/8.1只作补充
空间复现，不与9.2混合平均。

不能写：FMT 对任意 3D flow 都必然提高；Task3 已验证 streamwise/spanwise/hairpin
分类；7.2/8.1 达到了 `+.20` F1 扩展目标；development 搜索的 `+.2071` 是独立确认
结果。

## 合并来源

本文件取代以下分散文档：

- `mainExp_Task3Universality_1.1.md`
- `mainExp_Task3Universality_2.2.md`
- `mainExp_Task3NewFlows_2.3.md`
- `mainExp_Task3_3D_3.1.md`
- `mainExp_Task3_3D_3.2_global_ivd.md`
- `mainExp_Task3_3D_5.1.md`
- `mainExp_Task3_3D_5.2.md`
- `mainExp_Task3_3D_6.1.md`
- `mainExp_Task3_3D_7.1.md`
- `mainExp_Task3_3D_7.2.md`
- `mainExp_Task3_3D_8.1.md`

旧文档的逐命令与逐文件细节仍可由 Git 历史恢复；当前代码/config/结果路径以
`experiment_log.md` 对应实验行为准。
