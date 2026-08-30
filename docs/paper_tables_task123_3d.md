# 3D Task1–Task3 与 Task5 论文性能表

本页只合并已冻结 confirmation 结果，不重新选择任何 feature、VAE、checkpoint、
cluster 映射或阈值。Task2/Task3 当前机器结果分别位于
`outputs/mainExp_Task2_3D_4.1/` 和 `outputs/mainExp_Task3_3D_6.1/`。Task2
冻结协议见 `docs/mainExp_Task23_3D_4.1.md`，Task3 anchored FMT 协议与哈希见
`docs/mainExp_Task3_3D_6.1.md`；Task5 机器表位于
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

每个 physical family 用 development validation 为 FMT 冻结一个 VAE；同一 family
的 Raw 与 FMT 两臂共用结构、latent dimension、KL 权重、学习率、训练步数和随机
种子。Linear 表示无 hidden layer；MLP 是多层感知机（Multilayer Perceptron），
方括号内为 hidden widths。全部 KL 权重为 `1e-6`。

| Flow | 冻结 VAE / FMT feature | Raw+VAE F1 | FMT+VAE F1 | 配对 F1 增益 |
|---|---|---:|---:|---:|
| Channel observer | MLP `[128,64]`, latent 8 / all+kin4 | .0548±.0102 | .2243±.0533 | **+.1695±.0498** |
| Half-cylinder Re160 | MLP `[512,256]`, latent 64 / all+kin4 | .4523±.0040 | .4890±.0146 | **+.0367±.0183** |
| Half-cylinder Re640 | MLP `[512,256]`, latent 64 / all+kin4 | .3305±.0012 | .4361±.0145 | **+.1056±.0146** |
| Half-cylinder Re6400 | MLP `[512,256]`, latent 64 / all+kin4 | .4768±.0061 | .5868±.0111 | **+.1101±.0093** |
| Tangaroa | MLP `[128,64]`, latent 8 / real-neighbor | .6474±.1764 | .7705±.0170 | **+.1231±.1809** |
| Delta-wing resampled | Linear, latent 8 / all | .4274±.0274 | .8250±.0208 | **+.3976±.0269** |
| Delta-wing original LBM | Linear, latent 8 / all | .4619±.0513 | .8524±.0031 | **+.3905±.0538** |
| F-22 | MLP `[256,128]`, latent 16 / all+kin4 | .6670±.0152 | .5958±.0202 | **−.0712±.0297** |
| Boeing 747 | Linear, latent 4 / kin2 | .5801±.0129 | .8114±.0300 | **+.2313±.0366** |
| Smoke buoyancy | Linear, latent 16 / real+imag-neighbor | .5184±.1489 | .7163±.0055 | **+.1980±.1538** |
| **Dataset macro** | — | **.4617** | **.6308** | **+.1691** |
| **Family macro** | — | **.4760** | **.6373** | **+.1613** |

新空间 primitive population 上，9/10 条目、6/7 family 为正；dataset-macro
增益达到预注册 `+.15` 目标。F-22 是保留的反例。该 confirmation 改变空间采样
相位，但时间切片在历史实验中出现过，因此不称 sealed temporal test。

## Task3：监督 IVD 二分类

本表使用与 Task1/Task2 相同的 whole-field IVD p95 标签。Raw-PCA residual 是
训练集内主成分分析（Principal Component Analysis, PCA）得到的 Raw-only feature；
其维度、residual 网络和训练设置均与对应 FMT residual 相同。Average Precision
（AP）衡量按预测分数排序后的 precision-recall 性能。

| Flow | Raw-PCA F1 | Raw+FMT F1 | F1增益 | Raw-PCA AP | Raw+FMT AP | AP增益 |
|---|---:|---:|---:|---:|---:|---:|
| Channel observer | .1383±.0034 | .8407±.0042 | **+.7024±.0076** | .0865±.0037 | .9041±.0150 | **+.8176±.0113** |
| Half-cylinder Re160 | .7189±.0097 | .8266±.0065 | **+.1077±.0161** | .7815±.0300 | .9318±.0039 | **+.1503±.0339** |
| Half-cylinder Re640 | .6799±.0145 | .9006±.0044 | **+.2207±.0101** | .7518±.0095 | .9661±.0011 | **+.2143±.0084** |
| Half-cylinder Re6400 | .5570±.0091 | .7975±.0021 | **+.2405±.0070** | .5807±.0117 | .8888±.0033 | **+.3081±.0150** |
| Tangaroa | .7647±.0056 | .8471±.0048 | **+.0824±.0008** | .8273±.0048 | .9402±.0013 | **+.1129±.0035** |
| Delta-wing resampled | .8565±.0012 | .9142±.0021 | **+.0577±.0008** | .9516±.0021 | .9767±.0003 | **+.0251±.0019** |
| Delta-wing original LBM | .8539±.0015 | .9102±.0003 | **+.0563±.0018** | .9420±.0010 | .9767±.0002 | **+.0347±.0009** |
| F-22 | .8049±.0138 | .9173±.0022 | **+.1124±.0116** | .8609±.0069 | .9764±.0000 | **+.1155±.0069** |
| Boeing 747 | .7895±.0022 | .8676±.0021 | **+.0781±.0001** | .8658±.0015 | .9477±.0000 | **+.0819±.0015** |
| Smoke buoyancy | .7844±.0072 | .8330±.0065 | **+.0485±.0006** | .8782±.0057 | .9381±.0018 | **+.0599±.0075** |
| **Dataset macro** | **.6948** | **.8655** | **+.1707** | **.7526** | **.9447** | **+.1920** |
| **Family-macro gain** | — | — | **+.1815** | — | — | **+.2060** |

这是 `mainExp_Task3_3D_6.1` 的第四空间 primitive population 独立确认：22.1
选出的 feature、模型、checkpoint、阈值、residual scale、Raw normalization 和
train-only Raw-PCA transform 均在生成确认数据前冻结；6.1 不训练、不调参。F1 与
AP 均在 10/10 条目、7/7 family 为正；dataset-macro F1 增益 `+.1707` 同时达到
预注册主目标 `+.15` 和扩展目标 `+.17`。最小条目 F1 增益为 Smoke 的 `+.0485`，
因此宏平均结论并非只由 Channel 的大增益产生。

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
- Task2：`mainExp_Task2_3D_4.1`（Ibex V100；10条目×4个新空间相位切片×5 paired seeds；summary SHA-256 `06c7de3e…be8122`）。
- Task3：`mainExp_Task3_3D_6.1`（Ibex GTX 1080 Ti/P100 生成第四空间population，CPU冻结推理；10条目×2 paired seeds×2 arms；summary SHA-256 `f46a307a…fb756`，`per_run.csv` SHA-256 `bc5a61b9…ef591`）。
- Task5：`mainExp_Task5_3D_1.1`（Ibex V100；10条目×5训练seed×4 held-out confirmation时间片×9个未见尺度tuple；归档SHA-256 `6053ed15…c58ec`）。
