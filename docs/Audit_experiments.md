# FMT 实验与证据审计汇总

本文合并项目中的论文证据审计、数值一致性审计、参数搜索审计和本地存储审计。
审计的作用是验证“记录与原始逐次结果是否一致”，不负责替实验选择更好结果。
完整实验流水见 [`experiment_log.md`](experiment_log.md)，Ibex 作业流水见
[`ibex_run_registry.md`](ibex_run_registry.md)。

## 当前审计结论

| 范围 | 当前可用结论 | 审计状态 |
|---|---|---|
| Task1 | 单一`fmt_all+kin4→PCA8`在10个3D数据条目的平均F1为`.6014`，统一Raw-PCA2为`.4510`，增益`+.1504`；9/10为正，F-22为明确反例 | 460条development选择与500条confirmation记录已独立复算；新增普通DFT强对照为`.5847`，故FMT相对最强对照只提高`+.0167` |
| Task2 | 单一 `fmt_all+kin4 + VAE(l512_l64_b1e-6)` 在10个3D数据条目上的直接Raw+VAE/FMT+VAE F1 为 `.48844/.58404`，增益 `+.09560` | 6.2的100条记录已独立重算；新增189维train-only Raw-PCA同VAE为`.58036`，FMT只高`+.00368`，因此大增益主张只适用于协议定义的直接Raw输入 |
| Task3 | 单一 `aivd1w3_dft` 配方的 Raw-PCA/FMT F1 为 `.63984/.85834`，增益 `+.21849`；AP为`.68910/.92357`，增益`+.23447`；10/10条目、7/7 family、5/5 seed macro 均为正 | 9.2的100条记录与Raw-wide强对照已独立重算；监督增益保持，但组件消融不支持把该增益专门归因于时间Fourier |
| Task4 | 仍是 channel proxy-label 探索；1.1 因空间泄漏无效，1.2 为当前协议 | 不进入 Task1–3 论文主结论 |
| Task5 | 相对 Raw-PCA residual 的 dataset-macro F1/AP 增益 `+.0891/+.1116`；Re640 AP 略降 | 冻结未见尺度 confirmation；反例保留 |

截至 2026-09-02，未发现当前论文总表与完整逐次记录之间的数值矛盾。发现过的三类
文字矛盾已经修正：一是把 Task2-5.2/control 误称为 Task2-4.1 实跑结果；二是旧
Task3 文档仍把被 8.1 取代的结果称为“当前”；三是统一配置 6.2/9.2 完成后，部分
汇总文档仍把逐-family 5.2/8.1 称为论文主结果。

## 论文证据审计的历史修订

2026-08-23 的中间审计当时依据的是 Task1-2.1/2.2、Task2-2.3/2.4 和
Task3-2.2/2.3。它正确识别了以下风险：Task1 的 F-22 反例；Task2 必须比较同一
VAE；Task3 的 constrained-AP 选择显式编码了待证明增益；Raw-wide 不是同构
Raw-only residual 对照；local 11-voxel IVD 标签跨网格不具同一物理尺度。

后续实验对这些问题的处理如下：

| 2026-08-23 风险 | 后续处理 | 当前结果 |
|---|---|---|
| Task1 只是 Task2 附带 direct 结果 | 先建立Task1-2.1/2.2独立confirmation，再以Task1-4.1仅从既有候选中选择一套跨10条目不变的FMT/PCA配方；KMeans只在train拟合、映射由validation冻结 | 统一主表FMT/Raw F1 `.6014/.4510`，增益`+.1504`，9/10为正；旧逐-family最优`.6130`保留为补充，F-22反例仍保留 |
| Task2 Raw/FMT VAE 配方不完全受控 | 5.2先验证同一physical-family内两臂共用VAE；6.1随后只用development为全部10条目选择一套不变的FMT/VAE配方，6.2在独立confirmation上冻结复核 | 当前统一主表dataset-macro F1 gain `+.0956`；8/10、5/7、5/5为正；旧5.2的`+.2392`只作补充 |
| Task3 选择规则要求 FMT 增益 | 3.1 起各方法只按自身 validation AP 选择；alpha/threshold 只用 development | 去掉该偏置后，结论取决于标签和表示；后续 p95 与 anchored FMT confirmation 恢复稳定正增益 |
| 缺少同构 Raw 对照 | 增加 train-only Raw-PCA residual，与 FMT residual 同维度同结构 | 当前 Task3 只相对 Raw-PCA 报核心增益 |
| local-IVD 标签跨网格不统一 | Task1/2/3/5 统一为 whole-field IVD p95 | 3.2 以后均使用同一 p95 协议 |
| 新 flow family 太少 | 加入 Boeing 747 与 Smoke buoyancy | 当前共 10 条目、7 family |

因此，旧审计中的“尚需补实验”属于历史计划，不再当作当前待办；它提出的风险已按上表
逐项处理，未解决的边界（Task1 F-22、Task5 Re640 AP、Task4 无人工类型 GT）继续
保留。

## Task1/Task2/Task3 统一配置独立审计

该轮审计专门检查“每个任务只允许一套跨10个数据条目不变的FMT配方”。选择只读取
development；confirmation不再选择模型、特征、阈值或超参数。

| 任务 | 冻结配方与确认容量 | 独立审计结果 | 证据身份与清理 |
|---|---|---|---|
| Task1-4.1 | `fmt_all+kin4→PCA8`；500条confirmation记录 | PASS；Raw/FMT F1 `.45102/.60138`，增益`+.15036` | evidence archive SHA `acc33a56…07b0`；audit JSON SHA `12472a18…5da6` |
| Task2-6.2 | `fmt_all+kin4` 189D；同一VAE为`[512,256]→latent64`、KL权重`1e-6`；100条记录 | PASS；Raw/FMT F1 `.48844/.58404`，增益`+.09560` | remote audit SHA `cc1580ed…79f4`；无checkpoint产生，cleanup前后均为0 |
| Task3-9.2 | `aivd1w3_dft`与64D辅助分支；100条记录 | PASS；F1增益`+.21849`，AP增益`+.23447` | remote audit SHA `e8af36e4…7a8e`；cleanup精确删除100个临时checkpoint，剩余0 |

Task2-6.2 中 Boeing 747 和 F-22 的F1增益分别为`−.10090`和`−.06009`，因此当前
结论是“大多数数据条目提高”，不是“所有流场提高”。Task3-9.2 的F1与AP则均为
10/10条目正增益。

## 更强 baseline 与组件消融独立审计

`experiments/Audit_Task123_AdditionalEvidence_1_1.py` 不导入正式汇总器，分别从逐次
CSV重建宏平均、记录容量和冻结选择。

| 实验 | 审计容量 | 审计结果 | 主要证据边界 |
|---|---:|---|---|
| `Verify_Task123_StrongBaselines_1.1` | Task1/2/3=`100/200/100` | PASS；summary/audit SHA=`eb31f471…a9ef`/`fc32dcc7…5cbf` | Task1相对普通DFT只剩`+.016719`；Task2相对Raw-PCA189同VAE只剩`+.003675`；Task3相对Raw-wide仍为`+.224354` F1、`+.238260` AP |
| `Ablation_Task123_FMTComponents_1.1` | Task1/2/3=`350/350/300` | PASS；summary/audit SHA=`b66b6be2…9846`/`5c5a37c4…4312`；Task3两臂容量相等 | neighbor Fourier是Task1/2最大贡献；Task3的first-only几乎复现DFT full，故不能把Task3增益专门归因于时间Fourier |

强baseline链保留三项运行异常。第一，读取冻结JSON时误用`json.tool`第二位置参数，
导致Task2冻结manifest被覆盖；未启动数组立即hold，并由job `51219246`依据同一config
确定性重建，逐字段与来源审计哈希通过后才释放。第二，原summary job `51218266`因
Task1历史目录别名缺失而失败；修复只是把两个SHA完全相同的已审计JSON复制到config
声明的别名路径。第三，首次独立audit job `51220233`错误要求Adjusted Rand Index
非负，但该指标合法范围是`[-1,1]`；只修正审计器范围后，job `51220282`从未改写的
逐次CSV重算并PASS。三项均未改变模型、配方、数据或性能记录，失败job保留在registry。

## 数值一致性与证据身份审计

### Task2/Task3 核心增益

`Verify_Task23CoreGain_1.1` 是统一配置实验之前的历史审计。它不调用正式 summary
实现，直接读取 Task2-5.2 的100条 Raw/FMT 同VAE记录和 Task3-6.1 的40条
Raw-PCA/FMT记录，检查
`dataset × seed × arm` 唯一性并重算宏平均：

- Task2 F1 gain `+.2391553`；10/10 datasets、7/7 families、5/5 seeds 正；
- Task3-6.1 F1/AP gain `+.1706578/+.1920209`；10/10 datasets、7/7 families、
  2/2 seeds 正；
- 与正式 summary 最大差 `2.50e-16`；审计 JSON SHA-256 `909ee699…b4d1`。

### Task3 独立空间确认

| 版本 | 逐次记录 | 结果 | 独立审计 |
|---|---:|---|---|
| 6.1 | 40/40 | F1/AP gain `+.17066/+.19202` | 所有 macro 误差 `<1e-12` |
| 7.2 | 40/40 | `+.17522/+.20146` | 最大差 `2.78e-17`；audit SHA `cdadc4d6…6350` |
| 8.1 | 40/40 | `+.18005/+.19991` | 最大差 `1.11e-16`；audit SHA `eabc384a…edb3` |

8.1 的第一次 evaluate 因冻结包遗漏 residual 引用的 Raw checkpoint，在任何推理前
失败并产生 0 条性能记录。修复只从原始 20 个 Raw checkpoint 按 SHA-256 精确复制
依赖，不重训、不改模型。失败 job 与修复链均保留在 registry。

### 论文表历史一致性

`Verify_Task23PaperTableConsistency_1.1` 从 Task2 三组证据和 Task3 四组证据的 520 条
逐次记录重新配对汇总：

- 与七份正式 summary 的最大绝对差为 `2.22e-16`；
- 7.2 与 8.1 的 40 个模型身份和冻结 manifest 身份完全相同，可解释为同一方法在
  两套空间 population 的复现；
- 论文表 135 个显示值全部通过四位小数舍入检查，最大舍入差 `4.97e-5`；
- 4.1、6.1 与 7.2/8.1 使用不同方法或 population，不得混成 paired 提升曲线。

### Task3 参数搜索审计

所有已完成的大规模 Task3 参数数组均要求：数组 child 数、训练次数、paired arm
容量、selection、逐次 CSV、无模型归档和 SHA-256 完整；selector 与 evidence audit
严格晚于 array。主要结论是：

- 48.1 full-stack development gain `+.20401`；
- 52.1/54.1 portfolio development gain `+.20552`；
- 57–64 的连续辅助参数搜索只增加 `+.001597`，74.1/76.1 最终 development gain
  `+.207113`、FMT F1 `.890572`；
- 65–76 没有产生通过 zero-tolerance FMT guard 的新增配方；
- 77.1 数组被用户终止，75 个已完成 children 的 partial metrics 从未读取；79.1–110.1
  未启动，111.1/112.1 未提交。因此这些版本没有性能结论。

development 搜索的最高数值不能替代 6.1/7.2/8.1 或当前统一配置9.2 confirmation。
这些旧搜索也不能写成达到`.893` absolute FMT F1；当前9.2在独立confirmation上
确实达到F1 gain `+.21849`，但它来自后来冻结的单一任务级配方，而非旧搜索链的
development最高点。

## 本地存储审计（2026-08-29）

审计前仓库 16,428 个文件、25.00 GiB。主要占用如下：

| 类型/目录 | 数量 | 大小 | 内容 |
|---|---:|---:|---|
| `.npz` | 4,199 | 15.80 GiB | primitive、FMT feature、seed、IVD reference、valid mask、元数据 |
| `.pt/.pth/.ckpt` | 638 | 270.03 MiB | 各类训练 checkpoint |
| `tmp/` | 901 | 3.78 GiB | Ibex 传输压缩包和下载副本 |
| `.git/` | 2,084 | 4.21 GiB | Git 对象；约 4.16 GiB 为未打包对象 |

已删除 638 个 checkpoint 和三个只含 checkpoint 的压缩包，释放 303.94 MiB；清理后
模型文件为 0，工作区为 15,787 个文件、24.70 GiB。仍保留的主要可重算内容为
15.80 GiB `.npz`、若干 `tmp` 数据包以及 Git 历史中的约 3.67 GiB 大对象。

这些缓存没有在本次文档整理中删除：批量删除 `.npz` 会影响本地复算，Git 历史重写
又是独立的破坏性操作，均需用户单独授权。当前协议是 checkpoint 仅可在 Ibex 依赖链
期间临时存在，最终 CSV/JSON 写完后删除，不下载、不提交。

## 以后如何记录审计

1. 审计运行本身作为一行追加到 `experiment_log.md`，Ibex job 同时登记到 registry；
2. 审计 JSON/CSV 放在 `outputs/<audit_id>/`，不得放在 `docs/results/`；
3. 只有改变论文证据边界或发现矛盾时，才更新本文；
4. 审计脚本放在统一实验代码目录，不再新增根目录 `Audit_*.py`；
5. 历史失败、哈希不一致和取消作业不得删除或改写成成功。

## 合并来源

本文取代 `paper_evidence_audit_2026-08-23.md` 和
`storage_audit_2026-08-29.md`。Task3 各轮独立审计的核心结果也已从相应 Verify 文档
合并到本文。代码审查与第一性原理问题仍分别保留在 `code_review_2026-08-16.md` 和
`first_principles_analysis.md`，因为它们审查的是实现正确性，不是实验结果一致性。
