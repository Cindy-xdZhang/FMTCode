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

## 运行结果（2026-08-27）

适应性评测完整读取 30 个既有候选在已暴露 ordinal 3--5 上的结果，冻结为
`c13_physical_kinematic_only`：44D physical-time kinematic FMT、
`geometry_fmt` residual、auxiliary width 64、minimum-gain 选择。它在 3 个 ordinal、
F1/AP、matched Raw-PCA/strong Raw 共 12 个 development 比较中的最差增益为
`+.08956`，平均增益 `+.19884`。选择文件写入时 fresh cache 尚不存在，且记录
`fresh_source_cache_read=false`。

冻结后使用 seed 70--74 重训，并一次性打开 starts 85/96：

| 方法 | F1（mean±std） | Average Precision（mean±std） |
|---|---:|---:|
| Raw | .3827 ± .0124 | .3367 ± .0330 |
| Raw-wide | .4028 ± .0240 | .3592 ± .0647 |
| 44D Raw-PCA residual | .3984 ± .0116 | .3222 ± .0210 |
| **44D physical-kinematic FMT residual** | **.7400 ± .0213** | **.8906 ± .0189** |

因此 FMT 相对 matched Raw-PCA 的 F1/AP 增益为 **`+.34157/+.56845`**，
相对 stronger Raw 为 **`+.33723/+.53148`**，主判据通过。五个 seed 的四类配对
增益（F1/AP × Raw-PCA/strong Raw）全部为正；相对 Raw-PCA 的 paired 95% t interval
分别为 F1 `[+.30677,+.37637]`、AP `[+.52521,+.61169]`，辅助稳健性判据也通过。

6/6 个全新尺度 tuple 的 F1 与 AP 增益均为正：F1 增益范围
`+.30071` 至 `+.37969`，AP 增益范围 `+.52786` 至 `+.59035`。两个时间片分别保留
3840/3844 个 primitive，各尺度有效样本约 633--654；IVD-p95 正类比例为
7.578%/5.255%。CPU 与 CUDA 对首个 fresh slice 的 3840×44 FMT feature 逐位一致，
排除了训练/评测设备数值差异。

本次因远端代码上传等待用户显式授权，实际在本地 RTX 3090 24 GB 完成；没有登记为
Ibex job。它不改变先前 Re640 的冻结结果：Re640 继续使用 `c24_physical_log`，其
outer FMT−Raw-PCA F1/AP 为 `+.09838/+.12161`，相对 strong Raw 为
`+.15269/+.20305`。两种 Reynolds number 使用不同 family-specific FMT 配方，论文
表格必须明确列出，不能伪装成统一超参数。

证据 SHA-256：

- adaptive per-run：`d905bc43adbffdbc74c239549a2422c0b1d752007623f99b32b5b3702c64bf11`
- adaptive leaderboard：`658971b4cb8039811ebd2f06cca5726d1e2e86f361d20c7ef90ee33acb0a29d1`
- frozen selection：`28bb661d861ff92e5ea1cab50c5cb73240409763c84e82fe5455ac70520e7543`
- frozen training config：`94c65c59bc5e4f879d2a0cdebcd672b2f57787209cf7b8b5e6d1087f19364ddd`
- fresh per-run：`70ea0446cdc2bc1ba47cb5d5124773147bc519fda25fe3a9ca4f8922610c9f93`
- fresh per-scale：`060c7d70c4cd431d0b1b4c47d3c4f5a96a4f009463bc3400152dca1f03f3d4b5`
- fresh audit：`6eb8ace26bdb9c3e7a8e6cb6361a0c0f3713de96c5a75bd518aed59b35feabff`

实现 commits：`5ef5a7c`, `f417057`, `a081b44`。
