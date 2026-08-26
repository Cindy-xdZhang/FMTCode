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

## Ibex V100 confirmation 结果

10个数据条目、7个physical family、5个训练随机种子、每条目8个held-out起始时间均
完成。所有family在只看development-validation Average Precision后均选择
Raw-PCA residual作为强Raw基线。

| Flow | Raw F1 | Raw-PCA F1 | Raw+FMT F1 | FMT−Raw-PCA F1 | Raw AP | Raw-PCA AP | Raw+FMT AP | FMT−Raw-PCA AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Boeing 747 | .8322 | .8744 | .9038 | +.0294 | .9082 | .9397 | .9682 | +.0285 |
| Channel observer | .1241 | .5618 | .7974 | +.2356 | .1036 | .6727 | .8746 | +.2019 |
| Half-cylinder Re160 | .6358 | .6877 | .7670 | +.0793 | .6779 | .7574 | .8595 | +.1020 |
| Delta-wing original | .8406 | .8722 | .9203 | +.0481 | .9309 | .9487 | .9754 | +.0267 |
| Delta-wing resampled | .8435 | .9020 | .9335 | +.0315 | .9306 | .9635 | .9816 | +.0181 |
| F-22 | .8460 | .9172 | .8903 | **−.0269** | .9083 | .9533 | .9404 | **−.0129** |
| Half-cylinder Re640 | .6908 | .7549 | .7577 | +.0028 | .7598 | .8470 | .8421 | **−.0049** |
| Half-cylinder Re6400 | .5668 | .6714 | .7079 | +.0364 | .5960 | .7553 | .7770 | +.0218 |
| Smoke buoyancy | .7631 | .7915 | .8239 | +.0324 | .8449 | .8825 | .9124 | +.0300 |
| Tangaroa | .7442 | .7850 | .8187 | +.0337 | .7840 | .8545 | .8884 | +.0339 |
| **Dataset macro** | **.6887** | **.7818** | **.8320** | **+.0502** | **.7444** | **.8575** | **.9020** | **+.0445** |
| **Family macro** | **.6832** | **.7888** | **.8436** | **+.0548** | **.7368** | **.8636** | **.9127** | **+.0490** |

相对Raw-PCA，F1为9/10条目、6/7 family正；AP为8/10条目、6/7 family正。8个
明确正F1条目的5-seed paired bootstrap 95%区间均高于0；Re640 F1 `+.0028`的区间
跨0，F-22为稳定反例（5/5 seeds均为负）。因此本结果支持“FMT相对强Raw特征扩展在
当前多数3D flow和macro-average上提高监督IVD-p95识别”，不支持“每个flow都提高”。

机器表：`outputs/mainExp_Task3_3D_3.2_global_ivd/final_confirmation/`。归档
SHA-256为`583ec77ca1e3c355b86de50987cc4548d9a557c2a4aad0291358b10fe0a4c040`。
