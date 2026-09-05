# mainExp_Task2_3D_2.4_newflows

## 目的与 same-VAE 协议

本实验把 `LBM_3D/boeing747_808_2392.nc` 和 `SmokeBuoyancy80_239.nc`
作为两个新增 3D flow family 加入 Task2。核心比较保持为：

- `Raw pathline → VAE → latent → KMeans`
- `FMT(pathline) → 同一 VAE → latent → KMeans`

每个 family 只用开发 validation 比较四个候选 VAE，并以 **FMT validation winner**
冻结一个架构；该完全相同的 hidden layers、latent dimension、KL 权重、学习率、
optimizer steps 和训练 seeds 随后同时用于 Raw/FMT。输入和输出层宽度只随 representation
维度变化。四个 confirmation 时间片不参与 VAE、FMT block 或匿名 cluster 映射选择。
时间片和 pathline 参数与 `docs/mainExp_Task1_3D_2.2_newflows.md` 相同。

代码与配置：`experiments/Run_Task2_3D_Main.py`、
`config/mainExp_Task2_3D_2.4_newflows.yaml`。

## Ibex A100 confirmation 结果

| Flow | 同一 VAE | Raw+VAE F1 | FMT+VAE F1 | 配对 F1 增益 | FMT ARI | FMT NMI |
|---|---|---:|---:|---:|---:|---:|
| Boeing 747 | MLP[256,128], latent-16, β=1e-3 | .6194±.0425 | .8474±.0120 | **+.2280±.0307** | .8181 | .6680 |
| Smoke buoyancy | linear latent-2, β=1e-5 | .8090±.0029 | .7437±.0096 | **−.0653±.0100** | .7041 | .5172 |

Boeing 强支持 FMT 是更好的 VAE 输入；Smoke 稳定反对“所有流场都提高”。Smoke 的
负增益不是给 Raw 单独使用更强 VAE 造成的：两臂都使用 FMT validation 选中的
`linear latent-2`，且三个配对 seed 的最差增益为 `−.0739`。

合并 `mainExp_Task2_3D_2.3` 后，same-VAE 主表覆盖 10 个数据条目、7 个 family：
7/10 条目和 5/7 family 为正增益；条目平均增益 `+.1562`，family-macro 平均增益
`+.1185`。F-22 与 Smoke 是明确反例，Re160 近似持平。

Ibex array job `50814544[0-1]` 在 `gpu203-16-r` 的 NVIDIA A100-SXM4-80GB 上完成，
汇总 job 为 `50814546`；正式结果副本位于
`outputs/mainExp_Task2_3D_2.4_newflows_ibex_a100/`。本地预跑保留在
`outputs/mainExp_Task2_3D_2.4_newflows/`，不作为论文主数字。
