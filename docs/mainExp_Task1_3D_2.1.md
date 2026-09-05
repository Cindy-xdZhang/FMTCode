# mainExp_Task1_3D_2.1

## 论文问题

Task1 只检验：**3D pathline primitive 经无训练的 Fourier Motion Transformer
(FMT) 编码后，KMeans(k=2) 是否能以可量化准确率区分 IVD 涡区与非涡区。**
本实验不把参考系客观性作为验收条件。

## 冻结协议

- 八个数据条目、五个物理族；每个开发场 10 个时间片，最终另用 4 个新时间片。
- 开发片 1–6 拟合 train-only StandardScaler/PCA/KMeans；7–8 只选择每个物理族的
  FMT block 和 PCA 维数；9–10 只确定匿名簇 0/1 中哪个表示涡。
- 最终 4 个时间片不参与表示、PCA、KMeans 或簇编号选择。
- 标签固定为标准 3D IVD 的全场第 95 百分位；没有根据 FMT 输出重选 IVD 阈值。
- 主报 F1、IoU；同时报告不需要簇编号映射的 Adjusted Rand Index (ARI) 和
  Normalized Mutual Information (NMI)。KMeans 用 5 个 seed，每次 `n_init=20`。
- Raw baseline 的 PCA 维数也在开发验证片独立选择。

代码与配置：`experiments/Run_Task1_3D_Main.py`、`FMT_Utils/Task12Data_3D.py`、
`FMT_Utils/Task12Evaluation_3D.py`、`config/mainExp_Task1_3D_2.1.yaml`。

## Ibex A100 新时间片结果（论文主表）

| Flow | FMT config | FMT F1 | FMT IoU | FMT ARI | FMT NMI | Raw F1 | FMT−Raw F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| channel | chirality / PCA2 | .2613±.0012 | .1503 | .2260 | .0881 | .0655 | **+.1957** |
| half-cylinder Re160 | full+kinematic / PCA8 | .5964±.0005 | .4249 | .4827 | .3188 | .3831 | **+.2133** |
| half-cylinder Re640 | full+kinematic / PCA8 | .5692±.0010 | .3978 | .4637 | .3153 | .3354 | **+.2338** |
| half-cylinder Re6400 | full+kinematic / PCA8 | .5330±.0004 | .3634 | .4747 | .2836 | .4189 | **+.1141** |
| Tangaroa | full+kinematic / no PCA | .7447±.0000 | .5932 | .7023 | .5052 | .5878 | **+.1569** |
| delta-wing resampled | neighbor real-spectrum / PCA2 | .7451±.0000 | .5937 | .7187 | .5968 | .5724 | **+.1727** |
| delta-wing original LBM | neighbor real-spectrum / PCA2 | .7656±.0000 | .6202 | .7374 | .6093 | .5192 | **+.2464** |
| F-22 | full+kinematic / PCA2 | .3136±.0000 | .1860 | .2589 | .1305 | .6564 | **−.3427** |

八项 FMT F1 均值约 `.5661`、中位数约 `.5828`、范围 `.2613–.7656`；7/8 条目的
FMT F1 高于独立选择过 PCA 的 Raw baseline。F-22 是明确反例：FMT 聚类不是随机结果
(ARI `.2589`)，但准确率明显低于 Raw，不能写成“FMT 在所有流场优于 Raw”。

## 可写结论

在当前五个 3D flow family、八个数据条目和固定 IVD-p95 标签上，FMT+KMeans
能够产生非平凡的涡/非涡聚类；七项达到 F1 `.533–.766`，channel 和 F-22 分别只有
`.261` 和 `.314`。因此 Task1 的“能够做到一定准确率”得到跨流场证据，但方法强度有
明显物理族差异，F-22 必须保留为与 Raw 比较时的反例。

本地预跑设备为 NVIDIA GeForce RTX 3090，结果位于 `outputs/mainExp_Task1_3D_2.1/`。
Ibex A100-SXM4-80GB 独立复跑 job `50788282` 成功完成；论文主结果副本位于
`outputs/mainExp_Task1_3D_2.1_ibex_a100/`。
