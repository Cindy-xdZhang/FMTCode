# 3D Task1–Task3 与 Task5 论文性能表

本页只合并已冻结 confirmation 结果，不重新选择任何 feature、VAE、checkpoint、
cluster 映射或阈值。Task1–Task3 原始机器表位于
`outputs/paper_tables_task123_3d/`；Task5 机器表位于
`outputs/mainExp_Task5_3D_1.1_ibex_v100/outputs/mainExp_Task5_3D_1.1/final_confirmation/`。

## Task1：training-free FMT + KMeans

| Flow | FMT feature / PCA | FMT F1 | Raw F1 | FMT−Raw F1 | ARI | NMI |
|---|---|---:|---:|---:|---:|---:|
| Channel observer | fmt_chirality_all / PCA-2 | 0.2613±0.0012 | 0.0655 | **+0.1957** | 0.2260 | 0.0881 |
| Half-cylinder Re160 | fmt_all+kin4 / PCA-8 | 0.5964±0.0005 | 0.3831 | **+0.2133** | 0.4827 | 0.3188 |
| Half-cylinder Re640 | fmt_all+kin4 / PCA-8 | 0.5692±0.0010 | 0.3354 | **+0.2338** | 0.4637 | 0.3153 |
| Half-cylinder Re6400 | fmt_all+kin4 / PCA-8 | 0.5330±0.0004 | 0.4189 | **+0.1141** | 0.4747 | 0.2836 |
| Tangaroa | fmt_all+kin4 / no PCA | 0.7447±0.0000 | 0.5878 | **+0.1569** | 0.7023 | 0.5052 |
| Delta-wing resampled | fmt_real_neighbor / PCA-2 | 0.7451±0.0000 | 0.5724 | **+0.1727** | 0.7187 | 0.5968 |
| Delta-wing original LBM | fmt_real_neighbor / PCA-2 | 0.7656±0.0000 | 0.5192 | **+0.2464** | 0.7374 | 0.6093 |
| F-22 | fmt_all+kin4 / PCA-2 | 0.3136±0.0000 | 0.6564 | **-0.3427** | 0.2589 | 0.1305 |
| Boeing 747 | kin2 / PCA-2 | 0.8408±0.0002 | 0.4705 | **+0.3703** | 0.8109 | 0.6625 |
| Smoke buoyancy | fmt_all / PCA-2 | 0.7608±0.0000 | 0.5454 | **+0.2154** | 0.7253 | 0.5361 |

FMT 的条目平均 F1 为 `0.6130`；9/10 条目高于 Raw。

## Task2：Raw+VAE 与 FMT+同一 VAE

| Flow | 同一 VAE | Raw+VAE F1 | FMT+VAE F1 | 配对 F1 增益 | FMT ARI | FMT NMI |
|---|---|---:|---:|---:|---:|---:|
| Channel observer | linear8_b1e-5 | 0.0532±0.0003 | 0.2224±0.0570 | **+0.1693±0.0568** | 0.1773 | 0.0687 |
| Half-cylinder Re160 | mlp8_b1e-4 | 0.5548±0.0398 | 0.5535±0.0165 | **-0.0013±0.0549** | 0.4232 | 0.2828 |
| Half-cylinder Re640 | linear2_b1e-5 | 0.4725±0.0057 | 0.6709±0.0075 | **+0.1984±0.0034** | 0.6048 | 0.4273 |
| Half-cylinder Re6400 | linear2_b1e-5 | 0.5636±0.0064 | 0.6897±0.0127 | **+0.1261±0.0178** | 0.6391 | 0.4327 |
| Tangaroa | mlp16_b1e-3 | 0.7044±0.0135 | 0.7290±0.0095 | **+0.0246±0.0042** | 0.6713 | 0.4996 |
| Delta-wing resampled | mlp16_b1e-3 | 0.3133±0.0156 | 0.8207±0.0142 | **+0.5074±0.0283** | 0.7993 | 0.6795 |
| Delta-wing original LBM | mlp16_b1e-3 | 0.3030±0.1098 | 0.8306±0.0101 | **+0.5276±0.1139** | 0.8074 | 0.6801 |
| F-22 | mlp16_b1e-3 | 0.6297±0.0124 | 0.4772±0.0434 | **-0.1525±0.0453** | 0.4062 | 0.2196 |
| Boeing 747 | mlp16_b1e-3 | 0.6194±0.0425 | 0.8474±0.0120 | **+0.2280±0.0307** | 0.8181 | 0.6680 |
| Smoke buoyancy | linear2_b1e-5 | 0.8090±0.0029 | 0.7437±0.0096 | **-0.0653±0.0100** | 0.7041 | 0.5172 |

7/10 条目、5/7 family 为正；条目平均配对增益 `+0.1562`，family-macro `+0.1185`。

## Task3：监督 IVD 二分类

本表使用与 Task1/Task2 完全相同的 whole-field IVD p95 标签。Raw-PCA residual
是同结构、同特征维度、同可训练参数量的强 Raw-only 对照；其选择只读取
development-validation Average Precision，未读取 confirmation。

| Flow | Raw F1 | Raw-PCA F1 | Raw+FMT F1 | FMT−Raw-PCA F1 | Raw AP | Raw-PCA AP | Raw+FMT AP | FMT−Raw-PCA AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Boeing 747 | .8322 | .8744 | .9038 | **+.0294** | .9082 | .9397 | .9682 | **+.0285** |
| Channel observer | .1241 | .5618 | .7974 | **+.2356** | .1036 | .6727 | .8746 | **+.2019** |
| Half-cylinder Re160 | .6358 | .6877 | .7670 | **+.0793** | .6779 | .7574 | .8595 | **+.1020** |
| Delta-wing original LBM | .8406 | .8722 | .9203 | **+.0481** | .9309 | .9487 | .9754 | **+.0267** |
| Delta-wing resampled | .8435 | .9020 | .9335 | **+.0315** | .9306 | .9635 | .9816 | **+.0181** |
| F-22 | .8460 | .9172 | .8903 | **−.0269** | .9083 | .9533 | .9404 | **−.0129** |
| Half-cylinder Re640 | .6908 | .7549 | .7577 | **+.0028** | .7598 | .8470 | .8421 | **−.0049** |
| Half-cylinder Re6400 | .5668 | .6714 | .7079 | **+.0364** | .5960 | .7553 | .7770 | **+.0218** |
| Smoke buoyancy | .7631 | .7915 | .8239 | **+.0324** | .8449 | .8825 | .9124 | **+.0300** |
| Tangaroa | .7442 | .7850 | .8187 | **+.0337** | .7840 | .8545 | .8884 | **+.0339** |
| **Dataset macro** | **.6887** | **.7818** | **.8320** | **+.0502** | **.7444** | **.8575** | **.9020** | **+.0445** |
| **Family macro** | **.6832** | **.7888** | **.8436** | **+.0548** | **.7368** | **.8636** | **.9127** | **+.0490** |

相对 Raw-PCA residual，FMT 的 F1 在 9/10 条目、6/7 family 为正；Average
Precision 在 8/10 条目、6/7 family 为正。F-22 是稳定负例；Re640 的 F1
增益置信区间跨 0，且 Average Precision 略降。因此证据支持多数当前 3D flow
及 macro-average 上的提升，不支持“每个 flow 都提高”。

## Task5：不同尺度监督 IVD 二分类

Task5 在每个 primitive 中改变邻居距离、RK4 积分步长和积分步数，但固定输出
`7×32×3`。训练、validation 和 confirmation 的尺度 tuple 分别为 18、6、9 个，
彼此不重合；confirmation 的 4 个晚期时间片不参与模型或阈值选择。`strongest Raw`
由 physical-family development-validation Average Precision 冻结；Raw-PCA residual
与 FMT residual 均增加 268D 输入，网络结构和可训练参数量相同。

| Flow | fixed Task3 FMT transfer F1 | strongest Raw F1 | Task5 Raw-PCA F1 | Task5 FMT F1 | FMT−Raw-PCA F1 | Raw-PCA AP | FMT AP | FMT−Raw-PCA AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Boeing 747 | .7230 | .7148 | .7148 | .8065 | **+.0917** | .7822 | .8919 | **+.1097** |
| Channel observer | .2668 | .1303 | .1303 | .5538 | **+.4235** | .0893 | .6147 | **+.5254** |
| Half-cylinder Re160 | .4326 | .2419 | .2146 | .2389 | **+.0243** | .1298 | .1659 | **+.0361** |
| Delta-wing original LBM | .8298 | .7972 | .7972 | .8714 | **+.0741** | .8947 | .9503 | **+.0555** |
| Delta-wing resampled | .7345 | .8126 | .8126 | .8838 | **+.0713** | .8988 | .9560 | **+.0572** |
| F-22 | .3349 | .7402 | .7402 | .7578 | **+.0176** | .7789 | .7973 | **+.0184** |
| Half-cylinder Re640 | .4599 | .6552 | .6900 | .6930 | **+.0030** | .7717 | .7653 | **−.0063** |
| Half-cylinder Re6400 | .4134 | .4738 | .4861 | .5213 | **+.0352** | .4874 | .5346 | **+.0472** |
| Smoke buoyancy | .6871 | .6323 | .5802 | .6677 | **+.0875** | .5744 | .7392 | **+.1649** |
| Tangaroa | .3300 | .6795 | .6695 | .7325 | **+.0630** | .6561 | .7645 | **+.1084** |
| **Dataset macro** | **.5212** | **.5878** | **.5835** | **.6727** | **+.0891** | **.6063** | **.7180** | **+.1116** |
| **Family macro** | — | **.5941** | **.5862** | **.6972** | **+.1110** | **.6058** | **.7499** | **+.1441** |

相对同维度 Raw-PCA residual，FMT 的 F1 在 10/10 条目为正，Average Precision
在 9/10 条目为正；相对 development 冻结的 strongest Raw，F1/AP 均为 9/10
条目、7/7 family 为正。9/9 未见尺度 tuple 的宏平均 F1/AP 增益均为正，F1
增益范围 `+.0421` 到 `+.1242`。相对 fixed-scale Task3 FMT 直接迁移，Task5 FMT
的 dataset-macro F1 提高 `+.1515`，8/10 条目提高。限制是 Re640 的 AP 略降，
Re160 未超过 strongest Raw，Re160/Smoke 未超过 fixed-scale FMT transfer。

## 结果来源

- Task1：`mainExp_Task1_3D_2.1` + `mainExp_Task1_3D_2.2_newflows`。
- Task2：`mainExp_Task2_3D_2.3` + `mainExp_Task2_3D_2.4_newflows`。
- Task3：`mainExp_Task3_3D_3.2_global_ivd`（Ibex V100；10条目×5训练seed×8 held-out confirmation时间片）。
- Task5：`mainExp_Task5_3D_1.1`（Ibex V100；10条目×5训练seed×4 held-out confirmation时间片×9个未见尺度tuple；归档SHA-256 `6053ed15…c58ec`）。
