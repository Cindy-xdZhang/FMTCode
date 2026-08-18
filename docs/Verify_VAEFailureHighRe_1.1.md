# Verify_VAEFailureHighRe_1.1

## 问题与冻结协议

诊断 `Verify_Task2Universality_1.1` 中 FMT+VAE 在 half-cylinder Re640/Re6400
下降的原因。数据切分、FMT、IVD p95、VAE hidden layers、beta、optimizer steps 和
三个训练 seed 全部保持不变；只扫描 latent dimension `[4,8,16,32,64]`。标签只用于
held-out 两个 timeslices 的 KMeans F1，不参与 VAE 训练。

FMT 的 161 维输入由 23 维 center 和 `6×23` 维排序 neighbor 构成。每个 23 维 block
包含 6 个 real norm、6 个 imaginary norm、6 个 cosine 和 5 个 chirality triple
product。报告每个 block 的 MSE、相对目标能量的 normalized MSE 及 block-only direct F1。

## Latent dimension

| Dataset | latent 4 | latent 8 | latent 16 | latent 32 | latent 64 | direct FMT |
|---|---:|---:|---:|---:|---:|---:|
| Re160 | .202±.080 | .139±.088 | .213±.011 | .172±.089 | .152±.069 | .259 |
| Re640 | .192±.017 | .258±.075 | .284±.021 | **.332±.061** | .296±.055 | .496 |
| Re6400 | .207±.008 | .200±.005 | .203±.017 | **.210±.008** | .187±.017 | .556 |

增加容量能部分修复 Re640，但不能修复 Re6400。Re6400 的 total reconstruction MSE
从 latent 4 的 `.1100` 降到 latent 32 的 `.0367`，F1 只从 `.207` 变成 `.210`；因此
总体 MSE 下降不是判别信息保留的可靠代理。

## Feature-block reconstruction bias

baseline latent=16：

| Dataset | neighbor/center normalized-MSE ratio | hardest semantic block | hardest normalized MSE |
|---|---:|---|---:|
| Re160 | 2.06 | chirality | .223 |
| Re640 | 2.55 | chirality | .156 |
| Re6400 | 2.05 | chirality | .174 |

`StandardScaler` 后 neighbor slots 又乘以 `.5`，所以普通逐维 MSE 对单个 neighbor slot
的惩罚只有 center slot 的四分之一。三个场均实际出现 neighbor normalized MSE 明显
高于 center。chirality 在三个场中也始终是最难重构的语义 block。

Re6400 的 block-only direct F1 为：center `.548`、neighbor `.465`、real norm `.458`、
imaginary norm `.526`、cosine `.227`、chirality **`.580`**。最难重构的 chirality
恰好是该场最有判别力的 block。这是当前最直接的失败机制证据：全局 reconstruction
MSE 按总体样本和数值能量优化，未保护高 Reynolds number 涡区域依赖的旋向特征。

## 结论边界

本实验说明当前 VAE 失败由两部分组成：Re640 有明显容量因素；Re6400 主要不是容量
不足，而是 reconstruction objective 与聚类判别信息不一致。它尚未证明某种加权 loss
一定能修复问题。下一项因果实验应冻结 latent=16/32，对比 baseline MSE、neighbor
反权重 MSE 和 chirality-preserving MSE。

数据文件：`outputs/Verify_VAEFailureHighRe_1.1/{latent_runs.csv,block_reconstruction.csv,block_direct_f1.csv,summary.json}`。
