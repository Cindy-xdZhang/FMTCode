# 3D Task1–Task3 与 Task5 论文性能表

本页只合并已冻结 confirmation 结果，不重新选择任何 feature、VAE、checkpoint、
cluster 映射或阈值。Task2/Task3 当前机器结果分别位于
`outputs/mainExp_Task2_3D_5.2/` 和 `outputs/mainExp_Task3_3D_8.1/`。Task2
冻结协议见 `docs/mainExp_Task2_3D_5.2.md`；Task3 改进方法的两次独立空间确认见
`docs/mainExp_Task3_3D_7.2.md` 与 `docs/mainExp_Task3_3D_8.1.md`；Task5 机器表位于
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
| Channel observer | MLP `[128,64]`, latent 8 / all+kin4 | .0557±.0101 | .3050±.0428 | **+.2494±.0398** |
| Half-cylinder Re160 | MLP `[512,256]`, latent 64 / all+kin4 | .5496±.0048 | .6077±.0123 | **+.0581±.0112** |
| Half-cylinder Re640 | MLP `[512,256]`, latent 64 / all+kin4 | .3863±.0034 | .4873±.0365 | **+.1010±.0366** |
| Half-cylinder Re6400 | MLP `[512,256]`, latent 64 / all+kin4 | .4779±.0080 | .5900±.0086 | **+.1121±.0130** |
| Tangaroa | MLP `[128,64]`, latent 1 / real-neighbor | .2529±.0884 | .7269±.0812 | **+.4740±.1030** |
| Delta-wing resampled | Linear, latent 12 / all | .4580±.0247 | .8167±.0243 | **+.3587±.0434** |
| Delta-wing original LBM | Linear, latent 12 / all | .4975±.0220 | .8366±.0162 | **+.3392±.0335** |
| F-22 | MLP `[256,128]`, latent 1 / all+kin4 | .2192±.0761 | .4142±.1230 | **+.1951±.1581** |
| Boeing 747 | Linear, latent 6 / kin2 | .5151±.0431 | .8131±.0173 | **+.2979±.0490** |
| Smoke buoyancy | Linear, latent 24 / real+imag-neighbor | .5106±.1362 | .7167±.0065 | **+.2061±.1365** |
| **Dataset macro** | — | **.3923** | **.6314** | **+.2392** |
| **Family macro** | — | **.3575** | **.6235** | **+.2660** |

第五空间 primitive population 上，10/10 条目、7/7 family、5/5 seed macro
均为正；dataset-macro 增益同时达到预注册 `+.15` 和期望 `+.22`。作为同一数据和
seed上的诊断，Task2-4.1 latent control 为 Raw/FMT `.4859/.6522`、增益`+.1662`，
仍达到`.15`，但F-22为`−.0743`。selected FMT绝对F1比control低`.0207`；因此主表
证明的是冻结共享瓶颈下FMT输入优势扩大，不声称latent选择提高FMT绝对准确率。

## Task3：监督 IVD 二分类

本表使用与 Task1/Task2 相同的 whole-field IVD p95 标签。Raw-PCA residual 是
训练集内主成分分析（Principal Component Analysis, PCA）得到的 Raw-only feature；
其维度、residual 网络和训练设置均与对应 FMT residual 相同。Average Precision
（AP）衡量按预测分数排序后的 precision-recall 性能。

| Flow | Raw-PCA F1 | Raw+FMT F1 | F1增益 | Raw-PCA AP | Raw+FMT AP | AP增益 |
|---|---:|---:|---:|---:|---:|---:|
| Channel observer | .1182±.0065 | .8428±.0043 | **+.7246±.0109** | .0696±.0066 | .8847±.0095 | **+.8151±.0029** |
| Half-cylinder Re160 | .7090±.0207 | .8672±.0118 | **+.1582±.0325** | .7827±.0007 | .9532±.0073 | **+.1704±.0066** |
| Half-cylinder Re640 | .6739±.0005 | .8775±.0030 | **+.2036±.0035** | .7315±.0062 | .9490±.0054 | **+.2175±.0008** |
| Half-cylinder Re6400 | .5365±.0122 | .7768±.0004 | **+.2403±.0126** | .5649±.0056 | .8787±.0008 | **+.3138±.0064** |
| Tangaroa | .7494±.0102 | .8520±.0022 | **+.1026±.0079** | .7819±.0182 | .9305±.0061 | **+.1487±.0121** |
| Delta-wing resampled | .8865±.0006 | .9269±.0005 | **+.0404±.0011** | .9616±.0005 | .9832±.0002 | **+.0216±.0006** |
| Delta-wing original LBM | .8668±.0027 | .9031±.0021 | **+.0363±.0048** | .9491±.0010 | .9769±.0020 | **+.0278±.0010** |
| F-22 | .7997±.0077 | .9218±.0019 | **+.1221±.0059** | .8812±.0030 | .9824±.0005 | **+.1011±.0025** |
| Boeing 747 | .7623±.0029 | .8778±.0011 | **+.1155±.0040** | .8240±.0172 | .9440±.0004 | **+.1200±.0168** |
| Smoke buoyancy | .8107±.0032 | .8676±.0028 | **+.0569±.0060** | .8926±.0035 | .9558±.0014 | **+.0631±.0049** |
| **Dataset macro** | **.6913** | **.8714** | **+.1800** | **.7439** | **.9438** | **+.1999** |
| **Family-macro gain** | — | — | **+.1944** | — | — | **+.2152** |

这是 `mainExp_Task3_3D_8.1` 的第七空间 primitive population 独立确认。逐
physical-family 的 feature 与训练配方只在 development 数据上选择；40个模型、
阈值、residual scale、Raw normalization 和 train-only Raw-PCA transform 均在生成
确认数据前冻结。确认阶段不训练、不调参。F1 在10/10条目、7/7 family和2/2 paired
seeds均为正；最小条目增益为 Delta-wing original LBM 的`+.0363`，因此宏平均结论
并非只由 Channel 的大增益产生。

| 独立空间确认 | 方法状态 | Raw-PCA/FMT F1 | F1增益 | Raw-PCA/FMT AP | AP增益 |
|---|---|---:|---:|---:|---:|
| mainExp_Task3_3D_6.1，第四population | 早期 anchored FMT | .6948/.8655 | +.1707 | .7526/.9447 | +.1920 |
| mainExp_Task3_3D_7.2，第六population | 改进后的冻结portfolio | .6858/.8611 | +.1752 | .7362/.9377 | +.2015 |
| mainExp_Task3_3D_8.1，第七population | 同类扩展portfolio；当前主表 | .6913/.8714 | +.1800 | .7439/.9438 | +.1999 |

同一改进方法的7.2与8.1在两套未见空间population上的平均F1增益为`+.1776`，
平均AP增益为`+.2007`；两次都超过预注册`+.15`目标。6.1使用较早方法，作为独立
旁证保留，不与7.2/8.1合并成同一方法的平均值。

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

`Verify_Task23CoreGain_1.1` 使用 `Audit_Task23_CoreGain.py` 独立重算Task2逐次CSV，
得到F1增益`+.2391553`。Task3-7.2与Task3-8.1各有不导入正式汇总代码的独立审计器，
均从40条冻结推理记录重建dataset、family和seed宏平均；与正式汇总的最大差分别为
`2.78e-17`和`1.11e-16`。8.1的per-run、summary和audit SHA-256分别为
`78d19ec3…bc52`、`0066299c…07bd`和`eabc384a…edb3`。

- Task1：`mainExp_Task1_3D_2.1` + `mainExp_Task1_3D_2.2_newflows`。
- Task2：`mainExp_Task2_3D_5.2`（Ibex GTX 1080 Ti/P100；第五空间population，10条目×4切片×2 recipes×5 paired seeds；200/200唯一结果；summary/per-run SHA-256 `739b48ea…c3095`/`20d9de15…7111f`）。
- Task3：当前主表`mainExp_Task3_3D_8.1`（Ibex混合GPU生成第七空间population，CPU冻结推理；10条目×2 paired seeds×2 arms；summary SHA-256 `0066299c…07bd`，`per_run.csv` SHA-256 `78d19ec3…bc52`）；同方法独立复现`mainExp_Task3_3D_7.2`的summary/per-run SHA-256为`e89751f9…5248`/`7c1ecc37…3901`。
- Task5：`mainExp_Task5_3D_1.1`（Ibex V100；10条目×5训练seed×4 held-out confirmation时间片×9个未见尺度tuple；归档SHA-256 `6053ed15…c58ec`）。
