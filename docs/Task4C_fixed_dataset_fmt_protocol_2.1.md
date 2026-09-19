# mainExp_Task4C_FixedDatasetFMT_2.1：数据集 v2 上的固定中心 FMT（p35/h0、c156 × FPS6/FPS16）

数据：`mainExp_Task4C_FixedDataset_2.1`（r3，`data_audit.json` SHA-256 e5aaa4fb…；规则见 [Task4C_fixed_dataset_rules_v2.md](Task4C_fixed_dataset_rules_v2.md)）。`verify-source` 逐文件核对冻结哈希后才允许训练；数据只读。

## 从采样点到局部曲线簇（本版本的读取规则）

用户 2026-09-18 夜要求：对样本点用 FPS（farthest point sampling，最远点采样）取 6 个和 16 个邻居构建簇送入 FMT 网络。实现：

- 邻居候选池 = 同一流场中离该样本点最近的 64 个样本点（对全部样本的 `seeds` 建 KD-tree，不分折；只用几何，不用标签）。从中心点出发做 FPS，得到 16 个邻居的顺序；k = 6 取前 6 个（FPS 的贪心序列是嵌套的）。邻居表按流场缓存在 `outputs/mainExp_Task4C_FixedDatasetFMT_2.1/neighbors/`，只依赖冻结的 `seeds.npy`。
- 每个（样本，长度）对是一行：中心线取该样本在该长度的曲线，邻居线取邻居样本在**同一长度**的曲线。每样本三行，全部行都进入训练/评价。
- 簇按冻结的整束归一化（质心、最大半径）后交给冻结编码器 `FMT_Utils/Task4C_FPS16_1_1.py` / `Task4C_OriginalCenter_1_1.py`：它们收到的正是这 k 个邻居，内部的再选择是恒等映射（构建时断言）。p35/h0 的 141 维特征与 `transform_fmt`、c156 的 48 点重采样和加宽残差网络都不改。
- 划分：折 0 为测试；训练折（1–4）中每流场随机抽 10% 的样本（固定种子 98401+流场序号，一样本三行同进同出）作为验证，只用于选轮次；其余训练。行数 826,005 / 91,779 / 245,703（Channel 训练 133,871 样本、验证 14,875、测试 40,670；TBL 141,464 / 15,718 / 41,231）。

四臂：`p35_h0_fps6`、`p35_h0_fps16`（76,738 参数）、`c156_fps6`、`c156_fps16`（992,386 参数）；种子 96611/96612/96613；batch 128、AdamW 0.001/0.0001、Dropout 0.15、最多 500 轮、50 轮早停、验证 F1（平均精度打破平局）选轮、阈值 0.5、训练集逐特征标准化；模型只驻内存。

## 执行

本机 RTX 3090 运行 `python -m experiments.Task4C_FixedDatasetFMT_2_1 run-local`（verify-source → gpu-check → 12 次训练，3 个并发 → audit-results）。代码 commit `2e707ea5`（`identity` 记录 commit、配置与源码哈希、主机名）。冻结引擎的 V100 专用检查在本版本中去掉：簇在编码器之外用 float64 选定，编码器不再做最近邻取舍；其余确定性设置不变。实测每轮约 54 s（3 进程并发）。


代码：`FMT_Utils/Task4C_FixedDatasetFMT_2_1.py`（邻居表、划分、簇归一化、Dataset）、`experiments/Task4C_FixedDatasetFMT_2_1.py`（各阶段与本地链）、配置 `config/mainExp_Task4C_FixedDatasetFMT_2.1.json`、测试 `tests/test_task4c_fixed_dataset_fmt_2_1.py`（4 项通过）。

## 结果（2026-09-20，audit-results 通过；三种子均值 ± 样本标准差，折 0 测试，行级 = 样本×长度）

| 臂 | 参数量 | 验证 F1 | 测试 F1 | Channel / TBL | 短 / 中 / 长 | 测试平均精度 | 每次训练分钟 |
|---|---:|---:|---|---|---|---:|---:|
| p35_h0_fps6 | 76,738 | 0.7921 | **0.2423 ± 0.0173** | 0.2288 / 0.2545 | 0.272 / 0.233 / 0.221 | 0.340 | 364 |
| p35_h0_fps16 | 76,738 | 0.8280 | **0.2493 ± 0.0047** | 0.2365 / 0.2611 | 0.284 / 0.245 / 0.218 | 0.345 | 390 |
| c156_fps6 | 992,386 | 0.8398 | **0.2634 ± 0.0093** | 0.2365 / 0.2876 | 0.297 / 0.258 / 0.235 | 0.344 | 342 |
| c156_fps16 | 992,386 | 0.8707 | **0.2642 ± 0.0032** | 0.2433 / 0.2829 | 0.292 / 0.265 / 0.235 | 0.348 | 330 |

- 验证集（训练折内样本）F1 0.79–0.87，测试折（完全未见的 hairpin 实例）F1 只有 0.24–0.26：所有臂对未见实例几乎不迁移；c156 略优于 p35/h0，FPS16 略优于 FPS6，短长度曲线优于长曲线。测试正类平均概率约 0.19，验证正类约 0.70（p35/h0 FPS6 种子 96612 的分析）。
- 同一划分下 Ibex 的非 FMT baseline（[Task4C_fixed_dataset_baselines_protocol_2.1.md](Task4C_fixed_dataset_baselines_protocol_2.1.md)）验证 F1 0.93 但测试 F1 0.20–0.24，说明这是"按实例分折"的任务本身对所有方法都极难，不是 FMT 或簇构造的问题。
- 执行记录：12 次训练全部在本机 RTX 3090 完成（3 并发，p35/h0 每轮约 51 s、c156 约 85 s）；训练中途另一会话向本仓库提交文档使 HEAD 移动（2e707ea5 → 7702ee2a → 808dff59 等），索引 9 的启动校验因此失败一次；随后身份比对改为只比配置与冻结源码哈希（`same_code`，不比 git commit 与本驱动文件），以 `run-local --resume` 补跑索引 9–11，`summary.json` 记录出现过的四个 commit。冻结的训练循环与编码器未改。
- 证据：`outputs/mainExp_Task4C_FixedDatasetFMT_2.1/summary.json`（含逐流场、逐长度、逐实例召回）、`arms/<臂>/seed<种子>/final/.../{result.json,test_predictions.npz}`。
