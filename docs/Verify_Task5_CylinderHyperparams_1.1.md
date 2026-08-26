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
148,225。两个训练 seed 为 60、61；seed 60 在 A100，seed 61 在 V100，且每个候选
使用完全相同的设备分配，避免某个候选独占更强 GPU。

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
