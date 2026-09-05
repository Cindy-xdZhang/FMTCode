# 3D Tasks 论文性能表

本页只合并已冻结 confirmation 结果，不重新选择任何 feature、VAE、checkpoint、
cluster 映射或阈值。Task1、Task2、Task3 的论文主表分别使用一套跨全部10个数据
条目不变的任务级配置；Task5 固定一套 FMT encoder，仅按其预注册研究问题改变
pathline 尺度。Task2-5.2与Task3-8.1的逐physical-family配方只保留为历史补充证据，
不能与统一配置主表混合计算 macro。Task3 的完整方法演进见
`mainExp_Task3_3D.md`；Task5 机器表位于
`outputs/mainExp_Task5_3D_1.1_ibex_v100/outputs/mainExp_Task5_3D_1.1/final_confirmation/`。

## 当前论文主表状态

| 任务 | 当前主结果 | 证据边界 |
|---|---|---|
| Task1 | 单一`fmt_all+kin4→PCA8`的FMT+KMeans F1 `.6014`，统一Raw-PCA2为`.4510`，增益`+.1504`；9/10数据条目为正 | 支持一套training-free FMT配置能在多数当前3D流场中区分涡/非涡；F-22是反例。相对更强ordinary DFT的增益仅`+.0167`，必须一并报告 |
| Task2 | 单一`fmt_all+kin4`与单一same-VAE：F1 `.4884→.5840`，增益`+.0956`；8/10条目、5/7 family、5/5 seed为正 | 支持FMT改善直接Raw pathline的同一VAE输入；Boeing与F-22为反例。相对189维train-only Raw-PCA同VAE仅`+.0037`，因此主张不能扩展到所有强表示baseline |
| Task3 | 单一`aivd1w3_dft` residual：F1 `.6398→.8583`，增益`+.2185`；AP `.6891→.9236`，增益`+.2345`；10/10条目、7/7 family、5/5 seed为正 | 支持统一几何辅助表示提高监督IVD-p95二分类；Raw-wide仍落后，但组件消融表明不能把提升专门归因于时间Fourier |
| Task4 | 尚无可进入论文主性能表的人工标签 confirmation | 当前 channel 结果只有 proxy-label 探索 |
| Task5 | Raw-PCA/FMT F1 `.5835→.6727`，增益 `+.0891`；AP 增益 `+.1116` | 支持 variable-scale FMT；Re640 AP `−.0063` 为保留反例 |

## 任务级统一配置冻结状态

| 任务 | 论文主表 FMT 配方 | Pathline primitive 协议 | 当前状态 |
|---|---|---|---|
| Task1 | `fmt_all+kin4 → PCA8` | 7条线；RK4；`dt_scale=.25`；积分48步；每线重采样32点；邻居距离`0.5×`最小网格间距 | 已由4.1 development全局选择并通过独立confirmation审计 |
| Task2 | `fmt_all+kin4`（189D）→ MLP `[512,256]` VAE → latent-64；KL权重`1e-6` | 与Task1相同 | 6.1 development全局选择和6.2独立confirmation审计均`PASS` |
| Task3 | `aivd1w3_dft`（1D）→ 64D auxiliary residual；监督网络与Raw-PCA臂同结构、同容量 | 与Task1相同 | 9.1 development全局选择和9.2独立confirmation审计均`PASS` |
| Task5 | `all_plus_gram_kinematic`，268D | 7条线和每线32点固定；邻居距离、积分步长、积分步数按预注册尺度tuple变化 | 1.1统一主表已完成；逐half-cylinder配方只作补充，不重算主表macro |

Task1/2/3 的 pathline 配置完全相同：中心线加`x±/y±/z±`共7条线，RK4，
`offset_grid_scale=.5`、`dt_scale=.25`、积分48步、每线重采样32点。Task5仍固定
7条线、RK4和32个输出点，但在全部预注册集合中使用邻居距离`.25–1.25×`、积分
步长`.125–.30×`、积分步数`32–64`；相对Task1/2/3，范围分别为`.5–2.5×`、
`.5–1.2×`和`.67–1.33×`。Task5训练集合包含完全相同的`.5/.25/48` tuple，
因此固定尺度配置是其尺度范围中的一个明确锚点。

### 固定 pathline 参数联合敏感性

`Verify_Task123_FixedPathline_1.1`额外比较7套固定参数；每套参数分别完整运行
Task1、Task2、Task3，不是Task5式逐primitive混合尺度。development选择出的
`.25/48/48`只比当前`.25/48/32`增加积分后的重采样点数，物理轨迹不变。冻结选择后
在未见ordinals 8–9和5个新seed上的结果为：

| 指标 | 当前`.25/48/32` | 候选`.25/48/48` | 候选−当前 |
|---|---:|---:|---:|
| Task1 F1增益 | +.181547 | +.180536 | **−.001011** |
| Task2 F1增益 | +.074706 | +.079315 | **+.004608** |
| Task3 F1增益 | +.199758 | +.202823 | **+.003065** |
| Task3 AP增益 | +.220199 | +.227262 | **+.007063** |
| 三任务等权F1增益 | +.152004 | +.154224 | **+.002221** |
| 三任务平均absolute FMT F1 | .706314 | .703565 | **−.002749** |

因此48点候选对预注册的平均相对增益有很小但经独立confirmation复现的提高；它没有
逐任务一致获益，且absolute FMT F1整体略降。本页现有Task1–3主表继续使用32点，
本结果作为参数敏感性补充，不与各任务原confirmation population混合重算主表。
完整development七配置排名和审计边界见`Verify_experiments.md`；confirmation证据归档
SHA-256为`62ecd98e…bec8`。

## Task2/Task3 历史确认总览

下表汇总旧确认和同样本诊断对照，现统一归入历史补充证据。所有数值均由逐次
`per_run.csv` 重新计算；“正条目/family”按配对 F1 增益是否大于零计数。VAE是
变分自编码器（Variational Autoencoder）；Average Precision（AP）衡量按预测分数
排序后的precision-recall性能。

| Task2 证据 | 角色与评估数据 | Raw+VAE F1 | FMT+VAE F1 | F1增益 | Family-macro增益 | 正条目/family |
|---|---|---:|---:|---:|---:|---:|
| mainExp_Task2_3D_4.1 | 历史4.1确认；seeds 9080–9084 | .4617 | .6308 | **+.1691** | +.1613 | 9/10；6/7 |
| mainExp_Task2_3D_5.2/control | 4.1 VAE配方在第五population和seeds 9090–9094上的诊断复测 | .4859 | .6522 | **+.1662** | +.1568 | 9/10；6/7 |
| mainExp_Task2_3D_5.2/selected | 历史逐family配方；与上一行完全相同的population和seeds | .3923 | .6314 | **+.2392** | +.2660 | 10/10；7/7 |

| Task3 证据 | 方法与空间population | Raw-PCA/FMT F1 | F1增益 | Raw-PCA/FMT AP | AP增益 | 正条目/family |
|---|---|---:|---:|---:|---:|---:|
| mainExp_Task3_3D_4.1 | 早期FMT residual；4.1 population | .7141/.8141 | **+.1000** | .7834/.9024 | **+.1190** | 10/10；7/7 |
| mainExp_Task3_3D_6.1 | anchored FMT；第四population | .6948/.8655 | **+.1707** | .7526/.9447 | **+.1920** | 10/10；7/7 |
| mainExp_Task3_3D_7.2 | 当前冻结模型组合；第六population | .6858/.8611 | **+.1752** | .7362/.9377 | **+.2015** | 10/10；7/7 |
| mainExp_Task3_3D_8.1 | 同一冻结模型组合；第七population；历史逐family配方 | .6913/.8714 | **+.1800** | .7439/.9438 | **+.1999** | 10/10；7/7 |

这些行不是一条可直接比较的调参曲线：Task2-4.1与Task2-5.2使用不同空间
population和训练seed；只有5.2的control与selected是严格同数据、同seed对照。
Task3-7.2与8.1的40个`(dataset, seed, source)` checkpoint、feature、阈值和residual
scale身份逐项相同，因此可以作为同一方法的两次空间复现；4.1和6.1使用较早方法，
不得与7.2/8.1合并平均。

## Task1：training-free FMT + KMeans

论文主表固定一套跨全部10个数据条目不变的`fmt_all+kin4→PCA8`，Raw也固定为
一套全局Raw-PCA2。配方只由development ordinals 0–7选择；ordinals 8–9仅校准
匿名簇编号；四个confirmation时间片和5个新KMeans seed不参与选择。

| Flow | 统一FMT F1 | 统一Raw F1 | FMT−Raw F1 | FMT ARI | FMT NMI |
|---|---:|---:|---:|---:|---:|
| Channel observer | .1822±.0005 | .0657±.0001 | **+.1165** | .0963 | .0901 |
| Half-cylinder Re160 | .5958±.0000 | .3834±.0002 | **+.2125** | .4820 | .3182 |
| Half-cylinder Re640 | .5687±.0001 | .3356±.0006 | **+.2331** | .4631 | .3149 |
| Half-cylinder Re6400 | .5331±.0002 | .4190±.0003 | **+.1142** | .4748 | .2836 |
| Tangaroa | .7450±.0000 | .5453±.0000 | **+.1996** | .7024 | .5049 |
| Delta-wing resampled | .7687±.0000 | .5724±.0000 | **+.1962** | .7437 | .6270 |
| Delta-wing original LBM | .7468±.0000 | .5192±.0000 | **+.2276** | .7174 | .5896 |
| F-22 | .3105±.0000 | .6548±.0019 | **−.3443** | .2564 | .1299 |
| Boeing 747 | .8042±.0002 | .4704±.0004 | **+.3338** | .7636 | .5810 |
| Smoke buoyancy | .7589±.0002 | .5445±.0000 | **+.2144** | .7230 | .5336 |
| **Dataset macro** | **.6014** | **.4510** | **+.1504** | — | — |

独立审计从460条selection和500条confirmation记录重算，状态`PASS`，确认
9/10数据条目为正；F-22的`−.3443`必须保留。证据归档SHA-256为
`acc33a56…07b0`，独立审计SHA-256为`12472a18…5da6`。

### 补充表：逐physical-family最佳配置

下表是此前允许逐family选择时的历史结果，完整保留用于补充材料和可视化，不能再作为
论文Task1主性能表，也不能与上表混合计算macro。

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

逐family最优FMT的条目平均F1为`0.6130`；它比统一配置`.6014`高`.0116`，但使用了
不同flow family的不同超参数，因此只作为性能上界式补充结果。

## Task2：Raw+VAE 与 FMT+同一 VAE

论文主表固定`fmt_all+kin4`（189D）和同一个VAE：MLP hidden widths
`[512,256]`、latent dimension 64、KL权重`1e-6`、学习率`3e-4`、7000个
optimizer steps。Raw与FMT两臂在每个数据条目和seed上共享这些设置；唯一输入差异是
Raw pathline与FMT feature。配方由development ordinals 0–7和seeds 90–92全局选择，
confirmation使用未参与选择的ordinals 8–9与seeds 100–104。

| Flow | Raw+VAE F1 | FMT+VAE F1 | 配对 F1 增益 |
|---|---:|---:|---:|
| Channel observer | .1945±.0021 | .2292±.0032 | **+.0347** |
| Half-cylinder Re160 | .1982±.0023 | .2578±.0096 | **+.0596** |
| Half-cylinder Re640 | .3345±.0051 | .4271±.0092 | **+.0927** |
| Half-cylinder Re6400 | .3609±.0023 | .5869±.0019 | **+.2261** |
| Tangaroa | .7114±.0030 | .7608±.0094 | **+.0494** |
| Delta-wing resampled | .5241±.0440 | .8196±.0380 | **+.2955** |
| Delta-wing original LBM | .6348±.0096 | .7382±.0417 | **+.1034** |
| F-22 | .6209±.0075 | .5608±.0221 | **−.0601** |
| Boeing 747 | .8522±.0016 | .7513±.0189 | **−.1009** |
| Smoke buoyancy | .4530±.0156 | .7086±.0237 | **+.2556** |
| **Dataset macro** | **.4884** | **.5840** | **+.0956** |
| **Family-macro gain** | — | — | **+.0720** |

独立审计从100条逐次记录重算，状态`PASS`；8/10数据条目、5/7 family和5/5
training seed的配对F1增益为正。Boeing 747与F-22的负增益必须保留。统一配置增益
`+.0956`低于旧逐family配置的`+.2392`，但其结论更严格：一套完全不随流场变化的
FMT/VAE配方仍在多数数据条目上提高同一VAE。per-run、summary与远端独立审计
SHA-256分别为`e7381d4d…624c`、`a15d596a…0564`和`cc1580ed…79f4`；Task2不写
checkpoint，cleanup核验临时checkpoint数为0。

## Task2历史补充：逐physical-family Raw+VAE 与 FMT+同一 VAE

下表完整保留此前允许逐family选择时的代码与结果，用于补充材料和效果图，不再作为
论文统一配置主表。上节6.2结果才是当前论文主结果。

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
seed上的诊断，5.2 control复用4.1 VAE配方，得到Raw/FMT `.4859/.6522`、增益
`+.1662`，仍达到`.15`，但F-22为`−.0743`。这不是Task2-4.1历史结果；后者在旧
population上为`.4617/.6308`、增益`+.1691`。selected FMT绝对F1比同population
control低`.0207`；因此主表证明的是冻结共享瓶颈下FMT输入优势扩大，不声称latent
选择提高FMT绝对准确率。这里所有“主表”旧称均已撤销，数值角色固定为历史补充。

## Task3：监督 IVD 二分类

论文主表固定单一`aivd1w3_dft` FMT block（1D）和64D auxiliary residual，所有
数据条目使用同一个FMT配方。Raw-PCA与FMT residual两臂具有相同输入宽度、网络结构、
可训练参数量、训练预算、checkpoint和threshold规则。配方只由train ordinals 0–5、
validation ordinals 6–7与seeds 40–42选择；confirmation在ordinals 8–9上使用
paired seeds 40–44。

| Flow | Raw-PCA F1 | Raw+FMT F1 | F1增益 | Raw-PCA AP | Raw+FMT AP | AP增益 |
|---|---:|---:|---:|---:|---:|---:|
| Channel observer | .0936±.0099 | .7625±.0437 | **+.6689** | .0654±.0093 | .8495±.0276 | **+.7841** |
| Half-cylinder Re160 | .3449±.0266 | .7109±.0367 | **+.3659** | .3227±.0491 | .7028±.0907 | **+.3801** |
| Half-cylinder Re640 | .6830±.0152 | .8544±.0079 | **+.1714** | .7730±.0089 | .9379±.0099 | **+.1650** |
| Half-cylinder Re6400 | .5828±.0108 | .8893±.0069 | **+.3065** | .6135±.0212 | .9572±.0029 | **+.3437** |
| Tangaroa | .7368±.0203 | .9055±.0102 | **+.1687** | .7673±.0171 | .9747±.0060 | **+.2074** |
| Delta-wing resampled | .8434±.0033 | .9200±.0063 | **+.0767** | .9366±.0020 | .9826±.0009 | **+.0460** |
| Delta-wing original LBM | .8115±.0058 | .8950±.0054 | **+.0834** | .9025±.0035 | .9683±.0011 | **+.0658** |
| F-22 | .7880±.0119 | .8974±.0115 | **+.1094** | .8388±.0163 | .9626±.0077 | **+.1239** |
| Boeing 747 | .7895±.0012 | .8760±.0046 | **+.0865** | .8616±.0096 | .9582±.0017 | **+.0966** |
| Smoke buoyancy | .7248±.0167 | .8724±.0250 | **+.1476** | .8097±.0183 | .9418±.0178 | **+.1321** |
| **Dataset macro** | **.6398** | **.8583** | **+.2185** | **.6891** | **.9236** | **+.2345** |
| **Family-macro gain** | — | — | **+.2203** | — | — | **+.2423** |

独立审计从100条逐次记录重算，状态`PASS`；F1和AP均在10/10数据条目为正，F1在
7/7 family和5/5 paired seed为正。per-run、summary与远端独立审计SHA-256分别为
`6e80adfa…50a9`、`f577a632…6ab9`和`e8af36e4…7a8e`。审计前存在100个临时
checkpoint，cleanup已精确删除100个并确认剩余0。统一配方的F1增益`+.2185`高于旧
逐family 8.1的`+.1800`；但绝对FMT F1`.8583`低于8.1的`.8714`，两者使用不同
confirmation population和Raw对照，不能把差值解释为单独的feature改进。

## 补充表：更强非FMT baseline

这些对照使用与主表相同的冻结 confirmation，不替换上面的核心比较。普通离散Fourier
变换（Discrete Fourier Transform, DFT）直接对每条轨线的坐标序列变换，不使用FMT的
rotation-invariant Gram、chirality或sorted-neighbor构造。Raw-PCA189只在训练集拟合
主成分分析，将Raw pathline压到与FMT相同的189维，再进入完全相同的VAE。

| 任务 | 更强非FMT对照 | 对照F1（1.1） | 对照F1（1.2复现） | FMT F1 | FMT−对照 F1 | 解释 |
|---|---|---:|---:|---:|---:|---|
| Task1 | ordinary coordinate DFT，development冻结 | .5847 | .5843 | .6014 | **+.0167** | FMT略优于强Fourier对照；原始Raw-PCA2对比的+.1504仍保留，但不是最强对照 |
| Task1 | 曲率/扭率/速度统计（逐采样序列，development冻结） | — | .4471 | .6014 | **+.1543** | 简单微分几何量不足以替代FMT，与Raw-PCA2 `.4510` 相当；仅在1.2中运行 |
| Task2 | Raw-PCA189 + 同一VAE | .5804 | .5770 | .5840 | **+.0037** | 近乎打平；同种子（110–114）FMT回放`.5785`相对1.2对照仅`+.0015`；直接Raw+同一VAE的+.0956不能代表相对维度匹配控制的增益 |
| Task3 | Raw-wide监督网络 | .6340 | .6340 | .8583 | **+.2244** | 增加Raw网络容量未缩小差距；AP为`.6853→.9236`，增益`+.2383`；1.2为冻结模型推理复放 |

Task2的其他完整对照为：已搜索最强Raw专用VAE `.5397`（1.2 `.5407`）、ordinary coordinate DFT+
同一VAE `.5314`（1.2 `.5526`）、21对轨线pair-distance DFT+同一VAE `.2409`（1.2 `.2376`）。所有400条逐次记录由
独立审计重算，summary/audit SHA-256为`eb31f471…a9ef`/`fc32dcc7…5cbf`。1.2新种子复现
（`Verify_Task123_StrongBaselines_1.2`，450条记录，独立审计`PASS`，summary/audit SHA-256 `3e494ade…`/`448b7def…`）
确认三条结论稳定；Task2普通DFT两版相差`.021`，是同一VAE五种子macro的噪声量级。

## 补充表：FMT组件消融

Task1和Task2所有版本均保持189维；删除的组件以零替代，所以VAE输入宽度、网络、
latent dimension和训练预算不变。数值是相对各自full replay的dataset-macro F1变化。

| FMT版本 | Task1 F1 | Task1变化 | Task1变化（1.2） | Task2 F1 | Task2变化 | Task2变化（1.2） |
|---|---:|---:|---:|---:|---:|---:|
| Full | .6014 | — | —（1.2 full `.6014`） | .5855 | — | —（1.2 full `.5785`） |
| Without neighbor Fourier blocks | .2708 | **−.3306** | **−.3307** | .4718 | **−.1137** | **−.1067** |
| Without chirality | .5825 | −.0189 | −.0190 | .5489 | −.0365 | −.0231 |
| Without real/imag cosine | .5831 | −.0183 | −.0184 | .5509 | −.0346 | −.0302 |
| Without center Fourier block | .5956 | −.0058 | −.0058 | .5747 | −.0107 | +.0004 |
| Without local kinematic block | .5996 | −.0018 | −.0019 | .5748 | −.0106 | −.0120 |
| Kinematic block only | .2451 | −.3563 | −.3563 | .2956 | −.2898 | −.2784 |

Task3当前辅助表示的单独消融结果如下。`First only`和`First + early mean`不使用时间
Fourier变换；每个版本都与同宽、同容量Raw-PCA residual配对。

| Task3辅助表示 | FMT F1 | FMT AP | 相对DFT full的F1变化 | 相对DFT full的AP变化 | F1变化（1.2） | AP变化（1.2） |
|---|---:|---:|---:|---:|---:|---:|
| One-bin DFT full | .8586 | .9235 | — | — | —（1.2 `.8592`/`.9225`） | — |
| First only | .8583 | .9234 | −.0003 | −.0001 | −.0023 | −.0003 |
| First + early mean | .8553 | .9217 | −.0034 | −.0019 | −.0075 | −.0077 |

因此neighbor Fourier是Task1/Task2的主要组件；但Task3数据不支持把监督增益专门归因
于时间Fourier。1000条消融记录独立审计PASS，summary/audit SHA-256为
`b66b6be2…9846`/`5c5a37c4…4312`。1.2新种子复现（`Ablation_Task123_FMTComponents_1.2`，
1000条记录，独立审计`PASS`，SHA-256 `b2ff74fb…`/`c1de83f8…`）确认neighbor Fourier主导和Task3
结论稳定；chirality与cosine在两版均为负贡献，而center Fourier block在Task2由−.0107变为+.0004，
因此center Fourier与local kinematic block的Task2贡献在种子噪声内，不作为论文主张。

## 补充表：pathline 扰动鲁棒性

`Verify_Task123_NoiseRobustness_1.1/1.2`：所有模型、变换与阈值只在 clean 数据上拟合，只对
confirmation pathline 施加确定性扰动，全部臂看到逐位相同的扰动实现。高斯噪声 σ 以邻居偏移距离
（0.5×最小网格间距）为单位；丢帧保留端点并线性插回；截断保留前段再重采样为 32 点。表中为
dataset-macro F1，Task3 括号内为 Average Precision。

| 扰动 | Task1 Raw-PCA2 | Task1 普通DFT | Task1 FMT | Task2 Raw | Task2 Raw-PCA189 | Task2 FMT | Task3 Raw-PCA | Task3 Raw-wide | Task3 FMT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | .451 | .585 | **.601** | .490 | .580 | **.586** | .640 (.689) | .634 (.685) | **.859 (.924)** |
| 高斯 σ=.05 | .166 | **.360** | .173 | .386 | .109 | .153 | **.635** (.686) | .630 (.683) | .597 (**.757**) |
| 高斯 σ=.10 | .164 | **.289** | .134 | .267 | .109 | .152 | .594 (.665) | **.602 (.672)** | .436 (.655) |
| 丢帧 40% | .451 | .583 | **.606** | .492 | .316 | **.572** | .639 (.689) | .633 (.685) | **.856 (.925)** |
| 截断至 75% | .279 | **.581** | .567 | .535 | .417 | **.580** | .502 (.614) | .526 (.632) | **.783 (.926)** |
| 截断至 50% | .140 | **.512** | .282 | .564 | .305 | .338 | .353 (.449) | .372 (.469) | **.597 (.882)** |

FMT 对丢帧完全稳健，对截断的稳健性在 Task2/Task3 最好；对坐标高斯噪声则是最脆弱的表示之一：
Task1 普通 DFT 在每个 σ 都高于 FMT，Task3 三种 Raw 网络在 σ=.05 保留 99% 而 FMT 保留 70%。
完整 13 臂×10 条件表见 `outputs/Verify_Task123_NoiseRobustness_1.2/robustness_table.csv`；
两次实验独立审计均 `PASS`，summary SHA-256 `b9aa7d62…`（1.1）与 `683463e3…`（1.2）。

### 噪声类型扫描（`Verify_Task123_NoiseTypes_1.1` / `NoiseTypesStrong_1.1`）

预注册 22 个条件、13 个臂，全部报告。每格为 FMT F1 与最强非 FMT baseline F1（Task3 括号内为 AP）：

| 扰动（最强等级） | Task1 FMT / 最强baseline | Task2 FMT / 最强baseline | Task3 FMT / 最强baseline |
|---|---|---|---|
| 刚体旋转 90° | **.601** / .209 | **.585** / .173 | **.680 (.895)** / .188 (.209) |
| 邻居种子偏移 σ=.20 | **.602** / .589 | **.581** / .526 | **.801 (.880)** / .570 (.660) |
| 共模噪声 σ=.20 | .570 / **.585** | .326 / **.536** | **.854 (.925)** / .529 (.631) |
| 时间弯曲 50% | .371 / **.584** | .243 / **.520** | **.723 (.791)** / .600 (.656) |
| 高斯 σ=.03 | .206 / **.483** | .189 / **.488** | **.650 (.792)** / .639 (.689) |
| 平滑高斯 σ=.20 | .136 / **.233** | .152 / **.224** | .281 (.519) / **.458 (.599)** |
| 脉冲离群点 10% | .142 / **.309** | .151 / **.212** | .225 (.442) / **.438 (.585)** |

FMT 对坐标系与采样几何误差（旋转、种子偏移；Task3 下还有共模抖动、时间弯曲、σ≤.03 细高斯）明显更
鲁棒；对逐点独立噪声（白/平滑高斯、脉冲）是最脆弱的表示之一。完整 13 臂×23 条件表见
`outputs/Verify_Task123_NoiseTypesStrong_1.1/robustness_table.csv`；两次独立审计 `PASS`，summary
SHA-256 `efc8b0c8…`（A）与 `c23e1433…`（B）。

## Task3历史补充：逐physical-family监督 IVD 二分类

下表完整保留此前逐family选择的feature与训练配方，用于补充材料和效果图，不再作为
论文统一配置主表。上节9.2结果才是当前论文主结果。

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

`Verify_Task23CoreGain_1.1` 使用 `experiments/Audit_Task23_CoreGain.py` 独立重算Task2逐次CSV，
得到F1增益`+.2391553`。Task3-7.2与Task3-8.1各有不导入正式汇总代码的独立审计器，
均从40条冻结推理记录重建dataset、family和seed宏平均；与正式汇总的最大差分别为
`2.78e-17`和`1.11e-16`。8.1的per-run、summary和audit SHA-256分别为
`78d19ec3…bc52`、`0066299c…07bd`和`eabc384a…edb3`。

`Verify_Task23PaperTableConsistency_1.1` 使用
`experiments/Audit_Task23_PaperHistory.py` 对上述Task2三行和Task3四行共520条核心比较记录
重新配对汇总。逐数据集结果与七份正式summary的最大绝对差为`2.22e-16`；7.2与
8.1的40个逐次模型身份及冻结manifest身份完全相同；合并表135个显示值也全部
通过四位小数舍入检查，最大舍入差为`4.97e-5`。审计发现并修正了旧文档中
将5.2/control误称为Task2-4.1实跑结果、以及把历史Task3结果仍标作“当前”的文字
问题；未发现数值矛盾。

- Task1：当前统一主表`mainExp_Task1_3D_4.1`（10条目×4 confirmation时间片×5个新KMeans seeds；460条development选择与500条confirmation记录独立审计通过；证据归档SHA-256 `acc33a56…07b0`）；`mainExp_Task1_3D_2.1`与`mainExp_Task1_3D_2.2_newflows`只提供旧逐-family配方的补充结果。
- Task2统一主表：`mainExp_Task2_3D_6.2_uniform_confirmation`（14 A100、4 P100、2 V100 children；10条目×2 arms×5 seeds；100条逐次结果；20/20 stderr为空；summary/per-run/独立审计 SHA-256 `a15d596a…0564`/`e7381d4d…624c`/`cc1580ed…79f4`）。历史补充为`mainExp_Task2_3D_5.2`。
- Task3统一主表：`mainExp_Task3_3D_9.2_uniform_confirmation`（19 A100、1 RTX 2080 Ti children；10条目×2 arms×5 seeds；100条逐次结果；20/20 stderr为空；summary/per-run/独立审计 SHA-256 `f577a632…6ab9`/`6e80adfa…50a9`/`e8af36e4…7a8e`）。历史补充为`mainExp_Task3_3D_8.1`；同一逐family方法的另一空间复现为`mainExp_Task3_3D_7.2`。
- Task5：`mainExp_Task5_3D_1.1`（Ibex V100；10条目×5训练seed×4 held-out confirmation时间片×9个未见尺度tuple；归档SHA-256 `6053ed15…c58ec`）。
