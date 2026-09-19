# mainExp_Task4C_FixedDatasetBaselines_2.1：数据集 v2 上的非 FMT baseline

数据：`mainExp_Task4C_FixedDataset_2.1`（r3，`data_audit.json` SHA-256 e5aaa4fb…）。四个家族沿用 1.1 版本的冻结模型、编码、pilot 与训练代码（只重绑 `identity`）：Conv3D 16³/24³（72,192 参数）、BiLSTM+MLP（76,786）、PointNet（76,749）、PointNet++ SSG（76,723）。1.1 中的旋转中心 p35 参考不再运行（FMT 结果来自 `mainExp_Task4C_FixedDatasetFMT_2.1`）。

## 数据接口：把 v2 样本导出成这些引擎能读的线束文件

四个引擎读 v1 风格的 `physical/<flow>/<split>/{geometry.npy, seeds.npy, metadata.npz}` 与每流场 `preparation.json`。新增 `export` 阶段（CPU）从 v2 生成它们，簇与划分和 FMT 2.1 **完全相同**：

- 行 = （样本，长度），顺序与 FMT 2.1 一致；训练/验证/测试 = 折 1–4 去掉 10% 验证样本 / 该 10% / 折 0，行数 826,005 / 91,779 / 245,703。
- 每行 27 槽：槽 0 = 样本自身在该长度的曲线；槽 1–16 = FPS16 邻居（同流场 64 最近样本中的最远点采样顺序）在同一长度的曲线；槽 17–26 补零，`counts = 17`。整束按冻结归一化（质心、最大半径），播种点同框归一化写入 `seeds.npy`。
- `metadata.npz` 提供 v1 的 22 个键：`labels`、`instance`（v2 归属实例）、`head_component`（v2 候选分量）、`scale_id`（长度编号 0/1/2）、`center_number`（样本编号）等；引擎只读 `labels`、`counts`、`instance` 与 `preparation.json` 的文件哈希。
- 每流场 `preparation.json` 与 `export_<flow>.json` 记录文件哈希、行数、类别数、邻居表哈希与代码身份；各家族的 `reuse` 阶段逐文件核对后以只读符号链接挂到家族目录。

## 执行（Ibex）

部署目录 `/ibex/user/zhanx0o/FMT_Task4C_FixedDatasetBaselines_2p1_20260919`，checkout `78b3fe50`（分支 `codex/task4c_v2_baselines`）；v2 数据由本机 scp 上传，远端逐文件哈希与 `data_audit.json` 一致。提交 2026-09-19 UTC 凌晨，作业见 [ibex_run_registry.md](ibex_run_registry.md)（export 52070109_[0-1]；conv 52070110–52070114；bilstm 52070115–52070119；pointnet 52070120–52070124；pointnetpp 52070125–52070130）。每家族链：preflight → reuse → pilot/encode → train（种子 96611/96612/96613）→ summarize，全部依赖 export 成功；训练规则与 1.1 相同（batch 128、AdamW 1e-3/1e-4、Dropout 0.15、500 轮、50 轮早停、验证 F1 选轮、阈值 0.5）。

结果：`summarize` 写各家族 `summary.json`（合并测试 F1/AP、逐流场、逐长度、逐实例召回、三种子均值与样本标准差），完成后填入本文并汇入主表。

代码：`experiments/Task4C_FixedDatasetBaselines_2_1.py`（export / reuse / summarize / submit，其余阶段转调 1.1 引擎）、配置 `config/mainExp_Task4C_FixedDatasetBaselines_2.1.json`、启动器 `ibex_bash/task4c_fixed_dataset_baselines_2p1.sh`。

## 结果（2026-09-20，Conv3D / BiLSTM / PointNet 的 summarize 已完成；PointNet++ 三种子仍在训练，每轮约 14 分钟）

| 方法 | 参数量 | 测试 F1 | Channel / TBL | 短 / 中 / 长 | 测试平均精度 | 每次训练分钟 |
|---|---:|---|---|---|---:|---:|
| BiLSTM baseline2 | 76,786 | 0.240 ± 0.009 | 0.202 / 0.274 | 0.276 / 0.234 / 0.208 | 0.352 | 232 |
| Conv3D 16³ | 72,192 | 0.196 ± 0.010 | 0.156 / 0.230 | 0.231 / 0.197 / 0.157 | 0.347 | 365 |
| Conv3D 24³ | 72,192 | 0.208 ± 0.007 | 0.172 / 0.239 | 0.243 / 0.215 / 0.163 | 0.365 | 605 |
| PointNet small | 76,749 | 0.204 ± 0.017 | 0.178 / 0.225 | 0.233 / 0.208 / 0.166 | 0.317 | 569 |

验证 F1 均在 0.93–0.94（同实例样本），测试折（未见实例）0.20–0.24；与 FMT 2.1 的 0.24–0.26 同一水平。证据：远端 `outputs/mainExp_Task4C_FixedDatasetBaselines_2.1/{conv,bilstm,pointnet}/summary.json`。PointNet++ 结果待其 summarize（52070237）完成后补入。
