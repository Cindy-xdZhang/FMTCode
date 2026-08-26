# Verify_Task5_CylinderHyperparams_1.1：Re160/Re640 定向搜索预注册

## 目的

`mainExp_Task5_3D_1.1` 中 Re160 相对 matched Raw-PCA residual 的 F1/AP
增益仅 `+.0243/+.0361`，Re640 仅 `+.0030/-.0063`；Re160 的 FMT F1
还略低于 strongest Raw。这个验证实验只使用 Task5 development 数据，寻找对两个
Reynolds number 都稳定的 half-cylinder 配置。

已经看过的 `mainExp_Task5_3D_1.1` confirmation 不能再次充当独立确认集。因此本实验：

- development ordinal 0--2：训练；
- development ordinal 3：checkpoint、alpha 和候选选择；
- development ordinal 4--5：自动冻结候选后只打开一次的 outer development holdout；
- 旧 Task5 confirmation：本程序完全不读取。将来若复测，只能标为已见数据上的压力测试。

## 搜索内容

30 个候选同时覆盖：

1. 原 268D FMT 与每条 primitive 按真实时间校正的 kinematic FMT；
2. high-Re Task2 已显示有效的 neighbor real-spectrum、imaginary、chirality block；
3. time-local Gram、kinematic 及其组合；
4. `geometry_fmt`、`fmt_only`、`dual` 三种 residual 输入；
5. residual auxiliary width 32/64/96、learning rate `3e-4/1e-3/3e-3`；
6. fixed alpha、minimum-gain 和 constrained Average Precision 选择；
7. physical kinematic 的 signed-log 稳定化与 pseudoinverse cutoff。

每个 FMT 候选都配一个同输入维度、同 residual 网络、同训练预算的 Raw-PCA 候选。
Raw-PCA 仅在 ordinal 0--2 拟合。所有 residual 总参数必须小于 Raw-wide 的
148,225。两个训练 seed 为 60、61；原计划的 seed 60 A100 数组未启动，因预计等待至
次日而整体替换为 P6000，seed 61 使用 V100。每个候选在同一 seed 内使用完全相同的
设备分配，避免某个候选独占更强 GPU；设备替换未改变数据、模型、训练预算或选择规则。

## 冻结选择规则

候选只使用 ordinal 3 排序。对 Re160 和 Re640 分别计算 F1/AP 相对：

- 同维度 Raw-PCA residual；
- method-mean stronger(Raw, Raw-wide)。

主排序最大化上述 2 数据集 × 2 指标 × 2 baseline 的最小增益；并列时最大化平均增益。
选择代码写入 `selected_candidate.json` 后，outer 程序才允许读取 ordinal 4--5。

“明显增益”的预注册目标是：outer holdout 上两个 Reynolds number 的 F1 和 AP
相对 matched Raw-PCA 均至少 `+.03`，并且相对 stronger Raw 均为正。若未达到，
ordinal 4--5 不能被循环用于继续选参；该轮只能作为诊断，后续必须新增数据或改用新的
外层划分并明确标注适应性分析。

## 运行结果（2026-08-27）

30 个候选、2 个数据集、2 个训练 seed 的 120 个配对子作业全部成功，生成 240 个
candidate `per_run.csv`。selector 只读取 ordinal 3，冻结
`c24_physical_log`：268D `all_plus_gram_physical_kinematic`、physical kinematic
signed-log、`geometry_fmt` residual、auxiliary width 64、minimum-gain 选择。选择时
`outer_ordinals_were_read=false`；development 上的全局最差增益为 `+.09202`。

冻结后只打开一次 ordinal 4--5，结果为：

| Dataset | Raw-PCA F1 | FMT F1 | gain | Raw-PCA AP | FMT AP | gain | FMT−strong Raw F1/AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Re160 | .63100 | .64516 | +.01416 | .59564 | .63193 | +.03629 | +.02468 / −.03520 |
| Re640 | .67543 | .77380 | +.09838 | .73977 | .86138 | +.12161 | +.15269 / +.20305 |

因此预注册目标失败：Re640 获得明显且一致的增益；Re160 的 matched F1 增益低于
`+.03`，且 AP 未超过 stronger Raw。ordinal 4--5 自此属于已暴露 development
诊断，后续不得重复用作独立确认。

第一次 outer job `50907672` 因 checkpoint 文件名契约错误失败：评测器寻找
`fmt_residual`，训练器实际写入 `raw_fmt_residual`。它没有写出任何指标或 audit，
冻结候选也未改变。commit `aff6ce3` 将结果方法名与 checkpoint 文件名分开并增加
回归测试；同一冻结选择的重试 job `50909056` 成功。证据 SHA-256：

- `selected_candidate.json`: `ad7e26fcba5e52929080458d05be02c1603fa908c79b7f8cd1163ec82c0ec598`
- `validation_leaderboard.csv`: `d897162832b04aaa92d2a2f795c841ef9b766c6f07a6f8dc14451302396456e9`
- `outer_development_holdout/audit.json`: `c5807a356bf9cbb0683c7fea225a70a092bd3076f1b262c534c98c54d9418b5e`
- `outer_development_holdout/per_run.csv`: `5e5930faaaf501e5ec329cee2a5fa9acdf9eb91a6c2732e3be690b99d1c2e3cc`
