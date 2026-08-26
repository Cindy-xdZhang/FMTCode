# Verify_Task5_Re160FreshTimes_1.1：Re160 family-specific 配方与全新时间检验

## 原因与证据边界

`Verify_Task5_CylinderHyperparams_1.1` 的冻结 outer 结果显示 Re640 已获得明显增益，
但 Re160 相对 matched Raw-PCA residual 的 F1/AP 增益只有 `+.01416/+.03629`，
且 AP 相对 stronger Raw 为 `−.03520`。该负结果保留；development ordinal 3--5
均已暴露，今后只能用于 Re160 的适应性开发，不能再称为独立测试。

本实验允许 physical family 使用独立配置，但不改变数据或标签来制造优势。Re640 保持
冻结的 `c24_physical_log` 结果，不参与本轮选择。

## 冻结流程

1. 在全新缓存不存在时，对原搜索的 30 个候选统一评测 Re160 已暴露的 ordinal 3--5。
2. 对每个 ordinal 先平均 seed 60/61，再计算 F1、Average Precision 相对
   matched Raw-PCA 和 stronger(Raw, Raw-wide) 的增益。最大化 3 个 ordinal、2 个指标、
   2 个 baseline 的最小增益；并列时最大化平均增益。
3. 写入 `selected_re160_candidate.json` 和 `frozen_candidate_training.yaml` 后，才允许生成
   全新 source cache。最终在旧 development ordinal 0--4 训练、ordinal 5 选择
   checkpoint/threshold/alpha，使用新 seed 70--74。
4. 最终只打开一次新的 Re160 source starts 85、96。每个窗口读取 10 帧：旧 development
   最后窗口结束于 84，新窗口为 `[85,95)`、`[96,106)`，旧 confirmation 从 106 开始，
   三者不重叠。

## 全新尺度

最终测试使用 6 个原 train/validation/confirmation 都未出现的
`(offset_grid_scale, dt_scale, integration_steps)` tuple：

`(0.42,.18,36)`, `(0.42,.22,32)`, `(0.58,.16,40)`, `(0.58,.20,40)`,
`(0.90,.14,48)`, `(0.90,.25,32)`。

最大积分跨度为 8 source-frame；输入仍固定为 `7×32×3`。标签保持 whole-field IVD p95，
不因本次超参数探索改变。

## 对照与预注册判据

同一五个 seed 比较 Raw、参数更多的 Raw-wide、与 FMT 同维的 train-only Raw-PCA
residual、FMT residual。主判据为五 seed 平均：

- FMT 相对 matched Raw-PCA 的 F1 和 Average Precision 均至少 `+.03`；
- FMT 相对 stronger Raw 的 F1 和 Average Precision 均大于 0。

辅助稳健性判据要求至少 4/5 seed 的 matched F1/AP 配对增益为正。程序同时输出逐 seed、
逐尺度 tuple 指标；任何失败均保留，不得用 starts 85/96 继续选候选。

## 代码与输出

- workflow：`Verify_Task5_Re160FreshTimes.py`
- config：`config/Verify_Task5_Re160FreshTimes_1.1*.yaml`
- adaptive 输出：`adaptive_per_run.csv`、`adaptive_leaderboard.csv`、
  `selected_re160_candidate.json`
- fresh 输出：`fresh_test/per_run.csv`、`per_scale.csv`、
  `per_scale_summary.csv`、`audit.json`
