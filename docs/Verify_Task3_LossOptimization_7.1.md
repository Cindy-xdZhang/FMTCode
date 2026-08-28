# Verify_Task3_LossOptimization_7.1

## 问题

在不改变 Task3 主对照的前提下，测试类别不平衡损失和训练超参数能否进一步扩大
`FMT residual − train-only same-width Raw-PCA residual` 的 3D IVD 二分类增益。

## 冻结项

- 数据仅使用原 development、`mainExp_Task3_3D_4.1` 已公开训练空间 population、
  `mainExp_Task3_3D_5.1` 已公开 validation 空间 population。
- 不读取、生成或评估 `mainExp_Task3_3D_5.2` final population。
- 上游为 `Verify_Task3_SpatialRobust_5.2` Stage2 development selection。Ibex
preflight 在任何训练开始前记录该文件的 SHA-256；之后若文件改变，所有 child 拒绝运行。
基础 YAML 的身份哈希先将文本换行标准化为 LF，避免 Windows CRLF 与 Linux LF 使同一
Git 内容产生不同身份；运行时生成的 selection/manifest 仍使用原始字节 SHA-256。
- 使用 `Verify_Task3_NetworkArchitecture_6.1` 胜出的 two-hidden-layer 64-unit deep
  multilayer perceptron residual head。
- 每个候选对 FMT 与 Raw-PCA 两臂使用相同 optimizer、loss、batch size、training
  alpha、epoch 和 early-stop 设置；输入宽度及可训练 residual 参数量相同。
- paired seeds 固定为 40、41、42。

## 搜索空间

共 25 个候选：原设置 control、六个 positive-class weight scales、四个 focal-loss
gammas、两个 Dropout、三个 weight decay、两个 training alpha、cosine learning-rate
schedule、两个 batch size、延长训练，以及三个预先声明的组合候选。完整数值以
`config/Verify_Task3_LossOptimization_7.1.yaml` 为准。

Weighted binary cross entropy（加权二元交叉熵）是原始损失；focal loss 在其基础上
降低容易样本的梯度权重，使训练更集中于难分类样本。两种损失的正类权重都只由共同
训练标签的负/正样本比计算。

## 选择规则

允许每个 physical family 选择不同训练配方。主排序量为 development 数据上的
dataset-macro paired F1 gain；依次以 paired average-precision gain、正增益数据集数、
最差数据集 F1 gain 和最差 seed F1 gain 打破平局。开发目标为 `>= +0.15`，失败也必须
保留并记录。

## 规模与输出

- 10 datasets × 25 candidates = 250 Ibex array children。
- 每个 child 训练 3 seeds × 2 paired arms，总计 1500 次训练。
- preflight：`outputs/Verify_Task3_LossOptimization_7.1/preflight_manifest.json`
- leaderboard：`outputs/Verify_Task3_LossOptimization_7.1/optimization_leaderboard.csv`
- selection：`outputs/Verify_Task3_LossOptimization_7.1/optimization_selection.json`

## 当前状态

2026-08-28 本地静态预检和 Python 编译通过；系统 Python 没有安装 `pytest`，因此本地
使用直接数值断言验证：默认 weighted-BCE 与原 PyTorch 公式逐位相等，focal loss 有限
且能反向传播。Ibex 真实数据 preflight 将依赖上游 selector 完成后运行。
