# Ablation_Task23IVDPercentile_1.2

> 历史状态说明（2026-08-31）：下文“当前论文主表”指本消融实验冻结时的4.1
> 主表。现行主结果已更新为Task2-5.2和Task3-8.1；本消融没有用新模型重跑，
> 因此不能把4.1的绝对F1误写成5.2/8.1结果。

## 问题与固定协议

本实验复现当前论文主表 `mainExp_Task2_3D_4.1` 与 `mainExp_Task3_3D_4.1`，比较 whole-field IVD 的 p80、p85、p87.5、p90、p92.5 标签；原 p95 作为已发表主表参考，不与旧版 `Ablation_Task23IVDPercentile_1.1` 混用。

- 数据：当前 10 个 3D 条目、7 个 physical family，使用 4.1 的新空间 seeding phase 与 confirmation population。
- Task2：同一 physical family 内 Raw 与 FMT 使用同一个冻结 VAE。IVD 标签不参与 VAE 或 KMeans 训练，因此每个 dataset/method/seed 只训练一次，再用同一聚类预测对六套标签评分。
- Task3：每个请求百分位都重新训练 Raw、Raw-wide、同宽同结构 Raw-PCA residual 和 FMT residual；主比较为 FMT residual 对 Raw-PCA residual。
- 指标：10 个数据条目的 dataset-macro F1；Task3 同时报告 Average Precision（AP）。`seed-positive` 是 primitive seed 位置上的正标签比例，不是体素总体积比例。

## 完整结果

| IVD 标签 | seed-positive | Task2 Raw F1 | Task2 FMT F1 | Task2 增益 | Task2 正增益 | Task3 Raw-PCA F1 | Task3 FMT F1 | Task3 F1 增益 | Task3 F1 正增益 | Raw-PCA AP | FMT AP | Task3 AP 增益 | Task3 AP 正增益 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p80 | .242 | .5004 | .4414 | -.0590 | 5/10 | .7831 | .7940 | +.0109 | 6/10 | .8776 | .9057 | +.0282 | 7/10 |
| p85 | .181 | .4807 | .4977 | +.0169 | 5/10 | .7676 | .8132 | +.0455 | 8/10 | .8652 | .9123 | +.0470 | 8/10 |
| p87.5 | .152 | .4837 | .5347 | +.0510 | 5/10 | .7578 | .8197 | +.0619 | 10/10 | .8517 | .9143 | +.0626 | 9/10 |
| p90 | .122 | .4850 | .5751 | +.0901 | 6/10 | .7520 | .8195 | +.0675 | 10/10 | .8393 | .9113 | +.0720 | 10/10 |
| p92.5 | .094 | .4794 | .6131 | +.1337 | 7/10 | .7368 | .8151 | +.0783 | 10/10 | .8164 | .9101 | +.0937 | 10/10 |
| p95 reference | .066 | .4616 | .6303 | +.1687 | 9/10 | .7141 | .8141 | +.1000 | 10/10 | .7834 | .9024 | +.1190 | 10/10 |

请求扫描范围 p80–p92.5 内，p92.5 同时给出最大的 Task2 dataset/family-macro F1 增益、Task3 dataset/family-macro F1 增益和 Task3 dataset/family-macro AP 增益。若把原 p95 参考纳入排名，p95 的三项增益仍然最大。

## 解释

p92.5 相比 p95 将 seed 正标签比例从 .0657 扩大到 .0940，增加约 43%。它基本保持了 FMT 的绝对性能：Task3 FMT F1 从 .8141 变为 .8151，AP 从 .9024 变为 .9101；Task2 FMT F1 从 .6303 变为 .6131。相对增益变小的主要原因是更宽标签让 Raw 基线受益更多：Task3 Raw-PCA F1/AP 分别提高 .0227/.0330，而 FMT 只提高 .0010/.0077。

p92.5 的 Task3 F1 在 10/10 数据条目、7/7 family 为正，AP 也在 10/10、7/7 为正，因此不是只靠一个流场改变方向；但 Channel 的 F1/AP 增益 +.3694/+.4572 明显抬高宏平均，正文或附录仍应给逐流场表。Task2 p92.5 为 7/10 数据条目、5/7 family 正增益，负项是 F22、Re6400 和 SmokeBuoyancy。

这组结果支持把 p92.5 作为“更宽、视觉上较少把明显涡区标为负类”的候选折中值，但最大模型增益不能单独证明某个 IVD 百分位在物理上最正确。若论文保持原冻结协议，p95 应继续作为主表、p80–p92.5 作为敏感性分析；若后续依据可视化边界把 p92.5 改成新的主标签，应在新时间/空间 population 上再做一次不参与选择的确认。

## 项目协议决定（2026-08-28）

项目决定继续使用 whole-field IVD p95 作为当前 3D 论文的统一二分类标签：Task1/Task2 用于评估，Task3/Task5 用于监督。理由是 p95 在本实验完整扫描中给出最大的 Task2 F1 增益以及 Task3 F1、Average Precision 增益。p80–p92.5 保留为敏感性分析，不替换或重算已有 p95 主结果。

## 可复现证据

- 配置：`config/Ablation_Task23IVDPercentile_1.2.yaml`
- 汇总：`outputs/Ablation_Task23IVDPercentile_1.2/task23_ivd_percentile_4p1_table.{csv,md}`
- 逐流场：`outputs/Ablation_Task23IVDPercentile_1.2/task23_ivd_percentile_4p1_by_dataset.csv`
- Task3 完整性：500/500 development baseline 模型；50/50 final shards；1000/1000 final rows；所有训练与汇总 stderr 为空。
- 标签审计：40/40 个 4.1 source slices 的 p95 mask 与原缓存逐位一致。
- 结果归档 SHA-256：`abd5bb5ced7686b5f8afc6363c4d5504e6767400315f6ab19f3bf45cc224a1c9`
- 本地复算：四个汇总文件忽略 Windows/Unix 换行后逐字符相同；4/4 contract tests 通过。
