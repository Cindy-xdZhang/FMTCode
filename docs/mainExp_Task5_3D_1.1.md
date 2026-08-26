# mainExp_Task5_3D_1.1：3D 不同尺度几何学习

## 研究问题

Task5 将 Task3 的 fixed-scale pathline primitive 改为 variable-scale primitive。
IVD 标签、网络结构、FMT 268D 表示、Raw-PCA 268D 强对照、训练预算和模型选择规则
均沿用 `mainExp_Task3_3D_3.2_global_ivd`。唯一方法变化是每个 primitive 的：

1. 中心到 `x±/y±/z±` 邻居的初始距离；
2. RK4 数值积分步长；
3. 积分步数。

3D primitive 仍为中心加六邻居共7条线。每条线积分后统一重采样为32点，因此网络输入
恒为 `7×32×3`。用户描述中的5条线对应2D cross；在3D主实验中改成5条线会同时删除
一个空间轴，无法把结果只归因于尺度变化。

## 冻结的尺度拆分

- train：18个tuple，空间 offset 为`0.25/0.50/1.00`倍最小网格间距；6种
  `(dt_scale, integration_steps)`组合，总时间跨度为4–12个source frame。
- validation：6个未在train出现的tuple，offset为`0.375/0.750`。
- confirmation：9个train/validation均未出现的tuple，offset为
  `0.33/0.67/1.25`；其中`1.25`是空间尺度外推。
- 每个时间片内用固定随机种子打乱空间seed，再均衡分配尺度，任意tuple的分配数量差不超过1。

development使用4个训练时间片和2个validation时间片；confirmation使用4个更晚的
时间片。所有pathline source window的最大前视范围为12个source frame，development与
confirmation窗口不重叠。标签始终为`IVD(seed) >= percentile95(IVD volume)`。

## 论文主比较

1. fixed-scale Task3 Raw-PCA residual直接迁移到variable-scale confirmation；
2. fixed-scale Task3 Raw+FMT residual直接迁移；
3. Task5 variable-scale Raw与Raw-wide；
4. Task5 variable-scale Raw-PCA residual；
5. Task5 variable-scale Raw+FMT residual。

checkpoint仅按各自development-validation Average Precision选择，threshold仅用
development冻结，residual alpha固定为1.0。5个训练seed为`40–44`，便于与fixed-scale
Task3做paired比较。最终输出总体、逐流场、physical-family和9个confirmation尺度tuple
的F1与Average Precision。

## 代码与配置

- `FMT_Utils/MultiscalePathline_3D.py`
- `Build_Task5_Multiscale_Cache.py`
- `Evaluate_Task5_Multiscale.py`
- `config/mainExp_Task5_3D_1.1.yaml`
- `config/mainExp_Task5_3D_1.1_*.yaml`

## Ibex confirmation 结果

运行于 2026-08-26，10 个数据条目、5 个训练随机种子、每条目 4 个
confirmation 时间片均完成。9 个 confirmation 尺度 tuple 与 train/validation
tuple 完全不重合；confirmation 从未参与 checkpoint、threshold、Raw baseline 或
residual alpha 的选择。最终 primitive 形状逐样本固定为 `7×32×3`。

| Flow | fixed-scale Task3 FMT transfer F1 | Task5 Raw-PCA F1 | Task5 FMT F1 | FMT−Raw-PCA F1 | Raw-PCA AP | FMT AP | FMT−Raw-PCA AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Boeing 747 | .7230 | .7148 | .8065 | **+.0917** | .7822 | .8919 | **+.1097** |
| Channel observer | .2668 | .1303 | .5538 | **+.4235** | .0893 | .6147 | **+.5254** |
| Half-cylinder Re160 | .4326 | .2146 | .2389 | **+.0243** | .1298 | .1659 | **+.0361** |
| Delta-wing original LBM | .8298 | .7972 | .8714 | **+.0741** | .8947 | .9503 | **+.0555** |
| Delta-wing resampled | .7345 | .8126 | .8838 | **+.0713** | .8988 | .9560 | **+.0572** |
| F-22 | .3349 | .7402 | .7578 | **+.0176** | .7789 | .7973 | **+.0184** |
| Half-cylinder Re640 | .4599 | .6900 | .6930 | **+.0030** | .7717 | .7653 | **−.0063** |
| Half-cylinder Re6400 | .4134 | .4861 | .5213 | **+.0352** | .4874 | .5346 | **+.0472** |
| Smoke buoyancy | .6871 | .5802 | .6677 | **+.0875** | .5744 | .7392 | **+.1649** |
| Tangaroa | .3300 | .6695 | .7325 | **+.0630** | .6561 | .7645 | **+.1084** |
| **Dataset macro** | **.5212** | **.5835** | **.6727** | **+.0891** | **.6063** | **.7180** | **+.1116** |

相对同结构、同 268D 输入宽度的 Raw-PCA residual，FMT 的 F1 在 10/10
数据条目为正，Average Precision 在 9/10 为正；唯一 AP 反例是 Re640
`−.0063`。相对按 physical-family development-validation AP 冻结的 strongest
Raw baseline，dataset-macro F1/AP 增益为 `+.0849/+.1026`，数据条目方向均为
9/10，family 方向均为 7/7。相对只在固定尺度训练的 Task3 FMT 直接迁移，Task5
FMT 的 dataset-macro F1 提高 `+.1515`，8/10 数据条目提高；Re160 和 Smoke 是
两个 transfer 反例。

## Half-cylinder family-specific refinement（补充表）

原主表的统一 268D 配方保留不改。针对原表增益较小的 Re160/Re640，后续验证允许
half-cylinder physical family 使用 development-only 选择的独立 FMT 配方；标签、
网络、尺度分配原则和 Raw-PCA 匹配规则不变。两行使用不同的独立评测集，因此不与
原 10 条目表重新计算 macro：

| Flow | 冻结 FMT 配方 | 独立评测 | Raw-PCA F1 | FMT F1 | gain | Raw-PCA AP | FMT AP | gain |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Re160 | 44D physical-kinematic | fresh starts 85/96，6个全新尺度，5 seeds | .3984 | **.7400** | **+.3416** | .3222 | **.8906** | **+.5684** |
| Re640 | 268D physical-log | frozen development outer ordinals 4--5，2 seeds | .6754 | **.7738** | **+.0984** | .7398 | **.8614** | **+.1216** |

Re160 同时超过 strong Raw `+.3372/+.5315`，5/5 seed 和 6/6 全新尺度的 matched
F1/AP 增益均为正。Re640 同时超过 strong Raw `+.1527/+.2031`。这张补充表支持
“允许 physical-family 配方后，两种 Reynolds number 均有明显增益”，但不证明一个
统一超参数对二者都最优；Re640 数据长度不足以再构造与所有既有 source window
不重叠的新时间片，故其证据等级仍是首次冻结 outer，而非 Re160 式新时间检验。

完整协议、负结果和哈希见 `docs/Verify_Task5_CylinderHyperparams_1.1.md` 与
`docs/Verify_Task5_Re160FreshTimes_1.1.md`。

## 未见尺度分解

`o` 是邻居距离相对最小网格间距的倍数，`d` 是 source-frame 间隔的积分步长
倍数，`n` 是积分步数。下表对 10 个数据条目和 5 个训练随机种子取平均。

| Confirmation tuple | Raw-PCA F1 | FMT F1 | F1 gain | Raw-PCA AP | FMT AP | AP gain |
|---|---:|---:|---:|---:|---:|---:|
| `o=.33,d=.15,n=40` | .5846 | .6982 | **+.1137** | .6327 | .7685 | **+.1358** |
| `o=.33,d=.20,n=45` | .5580 | .6822 | **+.1242** | .5882 | .7359 | **+.1477** |
| `o=.33,d=.30,n=40` | .5755 | .6570 | **+.0815** | .5914 | .7113 | **+.1199** |
| `o=.67,d=.15,n=40` | .5848 | .6939 | **+.1090** | .6342 | .7584 | **+.1242** |
| `o=.67,d=.20,n=45` | .5813 | .6727 | **+.0914** | .6189 | .7196 | **+.1007** |
| `o=.67,d=.30,n=40` | .6060 | .6748 | **+.0688** | .6510 | .7445 | **+.0935** |
| `o=1.25,d=.15,n=40` | .5915 | .6902 | **+.0987** | .6402 | .7686 | **+.1284** |
| `o=1.25,d=.20,n=45` | .5704 | .6377 | **+.0672** | .6052 | .6901 | **+.0849** |
| `o=1.25,d=.30,n=40` | .6055 | .6477 | **+.0421** | .6221 | .7050 | **+.0828** |

9/9 未见尺度 tuple 的 F1 和 AP 平均增益均为正，包括训练范围外的
`offset_grid_scale=1.25`。但最宽空间尺度与最长积分跨度组合的 F1 增益降到
`+.0421`，因此当前结果支持所测范围内的跨尺度改善，不支持无限尺度外推。

## 证据与复现

- Ibex jobs：baseline `50892842[0-1]`、residual `50892843[0-19]`、merge
  `50892844`、confirmation `50892845`；全部 `COMPLETED`。
- 缓存：development 60 slices / 234,460 samples，confirmation 40 slices /
  155,680 samples；config SHA-256
  `e9eae4d2cc0a76ba768aed9a61cbbd430790109593cd4214cfaea93b76f56b4b`，
  cache SHA-256
  `1499358965063e8d87cd4c92a4e36cbb6726c42edf0ddbfae9eaaafb428412ce`。
- 机器表：`outputs/mainExp_Task5_3D_1.1_ibex_v100/outputs/mainExp_Task5_3D_1.1/final_confirmation/`。
- 结果与日志归档 SHA-256：
  `6053ed15e66b8f7438133fa92e7086bcd9b139e42a6a5cdfc6f7e682d86c58ec`。
