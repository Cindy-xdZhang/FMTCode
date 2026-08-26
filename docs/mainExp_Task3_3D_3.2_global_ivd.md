# mainExp_Task3_3D_3.2：统一 whole-field IVD p95 的 Task3

## 目的

本实验只改变 Task3 标签，使其与 3D Task1/Task2 完全相同：每个时间片先计算标准
Instantaneous Vorticity Deviation（IVD，瞬时涡量偏差），再以全体体素的第95百分位
作为阈值。监督标签为 `IVD(seed) >= percentile_95(IVD_volume)`。

## 固定比较

- Raw pathline；
- 参数更多的 Raw-wide；
- 冻结 Raw 主干 + 268D Raw-PCA residual；
- 冻结 Raw 主干 + 268D FMT residual。

模型、训练预算、数据划分、pathline、FMT 和 PCA 均沿用 `mainExp_Task3_3D_3.1`，只把
训练、验证、confirmation 标签替换为 whole-field IVD p95。随机种子改为 40–44。
主比较是 FMT residual 与同结构、同维度、同可训练参数量的 Raw-PCA residual。

标签由 `Build_Task3_GlobalIVD_Labels.py` 从 Task1/Task2 source cache 的 `reference`
逐位复制，不重新实现 IVD 或阈值计算。测试保证复制结果 bit-for-bit 相同。

## 证据边界

本轮 confirmation 使用 `mainExp_Task3_3D_3.1` 的八个未参与本轮训练/验证的起始
时间。它们此前已用于 Task1/Task2 及局部-IVD Task3 报告，因此是对本轮监督训练的
held-out confirmation，但不是从未被任何项目查看过的 sealed 数据。如果本轮开发结果
促使修改 FMT 模型，则这些时间片不得继续作为最终 confirmation，必须另取新时间片。
