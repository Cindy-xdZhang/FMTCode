# mainExp_Task1_3D_2.2_newflows

## 目的与冻结协议

本实验把 `LBM_3D/boeing747_808_2392.nc` 和 `SmokeBuoyancy80_239.nc`
作为两个新增 3D flow family 加入 Task1。方法、标签和选择流程继承
`mainExp_Task1_3D_2.1`：无训练 FMT feature 经 StandardScaler、可选 PCA 和
KMeans(k=2) 聚类；标签固定为全场 IVD 第 95 百分位。开发时间片只用于选择 FMT
feature block、PCA 维数以及匿名 cluster 到 vortex 的映射，四个 confirmation
时间片不参与任何选择。

- Boeing：开发 source indices `40,55,71,86,101,117,132,147,163,178`；
  confirmation `47,78,124,170`。
- Smoke：开发 source indices `32,44,57,69,81,94,106,118,131,143`；
  confirmation `38,63,100,137`。
- 两个数据集均使用 `16×16×16` seeding、RK4、48 个积分点、积分后采样 32 点，
  每个 confirmation 结果汇总 5 个 KMeans seeds。

代码与配置：`experiments/Run_Task1_3D_Main.py`、
`config/mainExp_Task123NewFlows_1.1_{development,confirmation}_cache.yaml`、
`config/mainExp_Task1_3D_2.2_newflows.yaml`。

## Ibex A100 confirmation 结果

| Flow | FMT config | FMT F1 | FMT IoU | FMT ARI | FMT NMI | Raw F1 | FMT−Raw F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Boeing 747 | kinematic-2 / PCA-2 | .8408±.0002 | .7254 | .8109 | .6625 | .4705 | **+.3703** |
| Smoke buoyancy | full FMT / PCA-2 | .7608±.0000 | .6139 | .7253 | .5361 | .5454 | **+.2154** |

两个新增 family 的 FMT+KMeans 都达到较高 F1，并明显高于独立选择 PCA 的 Raw
baseline。它们把 Task1 的确认范围从 8 个数据条目、5 个 family 扩展到 10 个数据
条目、7 个 family；合并旧结果后，FMT 在 9/10 个条目高于 Raw，F-22 仍是唯一反例。

Ibex job `50814543` 在 `gpu203-16-r` 的 NVIDIA A100-SXM4-80GB 上完成；正式结果副本位于
`outputs/mainExp_Task1_3D_2.2_newflows_ibex_a100/`。本地预跑结果保留在
`outputs/mainExp_Task1_3D_2.2_newflows/`，不作为论文主数字。
