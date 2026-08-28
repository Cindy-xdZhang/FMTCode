# Verify_Task3_NetworkArchitecture_6.1：Task3 配对网络结构搜索预注册

## 目的

回答一个限定问题：在数据、IVD-p95 标签、FMT 特征配方和训练预算固定时，哪一种网络结构 `X` 使 `X+FMT` 相对同结构 Raw-only 特征控制得到最大的跨流场平均 F1 增益。

主控制采用训练集拟合的同维 Raw-PCA，而不是故意削弱的低容量 Raw：

```text
frozen Raw backbone + architecture X(Raw-PCA)  vs
frozen Raw backbone + architecture X(FMT)
```

两臂的输入宽度、可训练参数量、初始化 seed、optimizer、epoch 上限、early stopping 和 alpha 网格完全相同。Raw-PCA 只在训练集拟合。

## 固定证据边界

- 基础时间 population：原 10 个 development slices，train ordinals `0–5`，validation ordinals `6–9`。
- 训练空间增强：已公开的 `mainExp_Task3_3D_4.1` 四个 slices。
- validation 空间增强：已公开的 `mainExp_Task3_3D_5.1` 四个 slices。
- 不读取 `mainExp_Task3_3D_5.2` 的最终 population，也不读取任何尚未公开的新时间片。
- FMT feature 由 `Verify_Task3_AnchoredRobust_5.1/stage2_selection.json` 固定；SHA-256 为 `8341272e…8a2ab`。
- base development config SHA-256 为 `c942fec5…1294f`。
- paired seeds：`40, 41, 42`。

## 搜索结构

共 7 种结构，全部使用 `geometry_fmt` 输入，auxiliary width 64：

1. 线性 residual head；
2. 历史浅层 MLP；
3. 两层深 MLP；
4. 两个 residual MLP blocks；
5. geometry/FMT 门控融合；
6. rank-32 低秩双线性交互；
7. 两 token、4 heads 的小型注意力融合。

每个结构必须低于冻结 Raw-wide 的总参数上限 `148,225`；超过者记录为无效，不允许放宽上限。

## 选择规则

对 10 个数据条目等权，首先最大化：

```text
mean_dataset(F1[X+FMT] - F1[X+Raw-PCA])
```

依次以 dataset-macro Average Precision 增益、正增益数据集数、最差数据集 F1 增益和最差 paired-seed F1 增益打破并列。该阶段只选择结构，不构成新 confirmation 结论；winner 冻结后才能登记新的未见空间 population。

## 代码与部署

- config：`config/Verify_Task3_NetworkArchitecture_6.1.yaml`
- runner/selector：`Search_Task3_NetworkArchitecture_6_1.py`
- CPU preflight：`ibex_bash/verify_task3_architecture_6.1_preflight.sh`
- GPU array：`ibex_bash/verify_task3_architecture_6.1_gpu.sh`
- selector：`ibex_bash/verify_task3_architecture_6.1_select.sh`
- 规模：`10 datasets × 7 architectures = 70` array children；每 child 内 `3 seeds × 2 paired arms = 6` 次训练。
- 总训练数：`420`；preflight 必须先验证全部真实缓存、标签、冻结 checkpoint、FMT 维度、双类别标签和参数预算，成功后才允许启动 GPU array。
