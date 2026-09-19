# Ablation_Task4C_BallQuery_2.2：固定 v2 的 ball query + FMT

2026-09-19 用户要求：仅使用一个种子；沿用冻结 v2 r3；ball query 初始半径 1h，不足 6/16 邻居则扩大半径，球内多于目标数才接最远点采样（farthest point sampling，FPS）。本版本只改变邻居选择，不搜索半径、不增加模型候选。

## 冻结条件

数据 `mainExp_Task4C_FixedDataset_2.1` r3，audit SHA-256 `e5aaa4fb8a8e01882c461da5916b3bb96883e2787b231b018b512d7e30d3a2a8`；Channel/TBL 189,416/198,413 样本；三种长度各成一行。训练/验证/测试 826,005/91,779/245,703 行，划分逐样本沿用 FMT 2.1：测试折 0，训练折内固定 10% 验证，不改变标签或重建曲线。

四臂 p35_h0_ball6 / p35_h0_ball16 / c156_ball6 / c156_ball16，均 **seed96611，单种子**。p35/h0 76,738 参数；c156 992,386 参数。141 维描述、原中心唯一 token、整束归一化、训练标准化、完整无卷积网络与旧 2.1 完全相同。每个邻居使用与中心同一目标长度的曲线。

训练仍为 batch128、AdamW 0.001/0.0001、dropout0.15、最多500轮、50轮早停、验证 F1 选轮（平均精度打破平局）、阈值0.5。只训练这四臂，不重跑旧 FPS、不增加种子；现有同 seed 旧 FPS 是比较证据，旧结果在 RTX3090、本轮在 V100，须注明设备差异，不宣称位级复现。

## 邻居规则

- **用户明确确认 `h_mode=global_min`**，即整个流场全部 x/y/z 相邻格点距离的最小值（每流场一个常数）：Channel `0.000016510486602783203`、TBL `0.007235100027173758`。执行修订 `r2_global_h_user_confirmed` 取代尚未部署/提交训练的局部h草案 `e0616ee5`；旧局部h诊断保留。配置记录原始网格哈希和 h 文件哈希；禁止直接将旧 `local_grid_scale`（三方向间距几何平均）误当最小间距。固定全局1h下全部零邻居；`h_definition_equivalence.json` 逐行核实两种 h 在本轮扩张规则下选出的邻居集合完全相同、均等于最近k个点且无第k距离并列，因此这一计量选择不影响本轮训练输入。
- 查询池保持旧实验定义：同流场全部样本的**播种点**，不按标签、实例或折过滤。跨折几何可见性保留且明确报告。
- 初始物理半径为 `1*h`。球排除中心自身；若少于 k，则半径增至第 k 个其他点的欧氏距离；这是本轮预定扩张规则，无人工调参。
- 若有效球内刚好 k 个，则全取；多于 k 个则以中心启动 FPS 取 k 个。精确距离并列按样本编号；边界用浮点一个单位的外向舍入与小量数值容差。每个结果保存初始数量、是否扩大、最终半径与选择编号。
- k6/k16 的扩半径独立计算；足够密集的原球具有相同 FPS 前缀，但稀疏区两个球不同，不能假称全局嵌套。
- 没有最大半径限制。因此自适应球仍可能跨越很远距离，不能保证邻居在同一个 hairpin 内。密度低时方案等于最近 k 个点，与旧“64最近点内FPS”仍不同。此限制须与结果一起报告。

## 核验、部署与输出

源码：`FMT_Utils/Task4C_BallQuery_2_2.py`（查询、缓存、复用原 Dataset）；`experiments/Task4C_BallQuery_2_2.py`（复用冻结训练与独立结果复算）；启动器 `ibex_bash/task4c_ball_query_2p2.sh`。

提交链：verify-source → gpu-check → train[0–3]%4 → audit-results。核验原数据六文件/冻结audit哈希、h文件哈希、样本划分；预检比较跨batch编码、真实小样本拟合、完整网络/参数/无卷积并记录GPU峰值内存。小样本拟合只作工程检查。Slurm作业跨节点，主机名只写runtime，不放入必须跨阶段相等的科学身份（旧本机驱动含host，不能原样用于集群）。

正式执行输出根 `outputs/Ablation_Task4C_BallQuery_2.2/r2_global_h/`；`source_verification.json` 记录球统计，`neighbors/*.npz` 保存全部邻居证据；`arms/*/seed96611/final/*/seed96611/` 保存选轮锁、日志和预测；`summary.json` 只有预测复算通过后才算正式结果。科学commit与实际作业号记于独立部署状态，避免提交后改动作为科学身份一部分的本文。

| 版本 | 技术与主代码 | 结果 |
|---|---|---|
| mainExp_Task4C_FixedDatasetFMT_2.1 | 冻结基线：最近64点→FPS6/16；FixedDatasetFMT_2_1.py | seed96611 p35/h0 测试F1 0.241171/0.246454，独立预测复算通过；其他历史种子不混入本轮 |
| Ablation_Task4C_BallQuery_2.2 | 1h球，不足扩至第k近点，必要时球内FPS；BallQuery_2_2.py | 部署与训练状态见 deployment_status.json；未完成不得填正式F1 |
