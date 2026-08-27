# Task2 / Task3 3D family-specific search 与最终确认（4.1）

> 状态：Task2-4.1 仍是当前 Task2 主结果；本页的 Task3-4.1 已被
> `mainExp_Task3_3D_5.1` 的 anchored FMT 新空间确认取代。旧数值保留用于追踪，
> 不再作为当前 Task3 论文主表。

## 结论

- **Task2 达到预注册目标。**同一变分自编码器（Variational Autoencoder, VAE）内，`FMT+VAE − Raw+VAE` 的 dataset-macro F1 为 `+0.16912`，高于目标 `+0.15`；9/10 个数据条目、6/7 个物理 family 为正。
- **Task3 提升方向成立，但没有达到预注册幅度。**相对同宽、同结构、同训练过程的 Raw-PCA residual，FMT residual 的 dataset-macro F1 / Average Precision 增益为 `+0.10000 / +0.11896`；两项均为 10/10 个数据条目、7/7 个 family 正增益，但 F1 未达到 `+0.15`。
- 两个结论都来自配方冻结后构建的新空间 primitive population。所用时间切片在历史实验中出现过，所以本实验不是全项目从未查看的 sealed temporal test。

## 冻结与确认协议

1. 10 个数据条目合并为 7 个物理 family：channel、half-cylinder、Tangaroa、delta-wing、F-22、Boeing 747、Smoke buoyancy。Re160、Re640、Re6400 共用 half-cylinder 配方；两种 delta-wing 共用配方。
2. 开发阶段只使用 ordinal 0–7：0–5 训练，6–7 选择。配方冻结并写入 SHA-256 后，才打开 ordinal 8–9 做 outer-development 审计。
3. Task2 每个 family 先从 14 种 FMT feature 与 4 种代表 VAE 中筛选，再将前三种 feature 与 12 种 VAE 交叉。每个候选内 Raw 与 FMT 两臂使用相同 hidden layers、latent dimension、KL 权重、学习率、optimizer steps 和随机种子。
4. Task3 先比较 14 种 FMT feature block，再将每个 family 的前三种 feature 与 10 种 residual route、宽度、学习率及融合选择组合交叉。主对照 Raw-PCA 只在训练数据上拟合，维度等于对应 FMT 输入，并使用完全相同的 residual 网络。
5. 最终确认前，将空间 seed-grid phase 固定为 `[0.31, -0.23, 0.17]`，重建 10×4 个切片，共 151,760 个有效 primitive。reference label 与 whole-field IVD p95 标签逐位相同。
6. Task2 使用 5 个新训练 seed `9080–9084`；Task3 使用 5 个 paired seed `40–44`。最终确认不再选 feature、网络、epoch、阈值、融合权重或簇编号。

冻结文件：

- Task2 selection SHA-256：`439d9e5dcf72adc78d12c38dda55443d87ffed2b8aeb061067b3cda8a8d97795`
- Task3 selection SHA-256：`9eb044145b9ea3781cd1c5a7ecfe4825c2dd1e2d9cc7e6db3e940ef92846122c`
- confirmation package SHA-256：`050cafd6b5a52d2e598dff924a1b29a25db92d21729bb07412fc625f045c0767`

## Task2：同一 VAE 的输入表示比较

线性 VAE 没有 hidden layer：输入分别直接映射到 latent mean / log-variance，latent 再经一个线性层重构输入。多层感知机 VAE 使用表中 hidden widths 和 GELU 激活，decoder 对称反向。所有冻结配方的 KL 权重均为 `1e-6`；精确学习率和训练步数见 `config/Verify_Task2_FMTVAEFamilySearch_4.1.yaml`。

| Flow | 冻结的同一 VAE / FMT 输入 | Raw+VAE F1 | FMT+VAE F1 | 配对增益 |
|---|---|---:|---:|---:|
| Channel observer | MLP `[128,64]`, latent 8 / `fmt_all+kin4` | .0548±.0102 | .2243±.0533 | **+.1695±.0498** |
| Half-cylinder Re160 | MLP `[512,256]`, latent 64 / `fmt_all+kin4` | .4523±.0040 | .4890±.0146 | **+.0367±.0183** |
| Half-cylinder Re640 | MLP `[512,256]`, latent 64 / `fmt_all+kin4` | .3305±.0012 | .4361±.0145 | **+.1056±.0146** |
| Half-cylinder Re6400 | MLP `[512,256]`, latent 64 / `fmt_all+kin4` | .4768±.0061 | .5868±.0111 | **+.1101±.0093** |
| Tangaroa | MLP `[128,64]`, latent 8 / `fmt_real_neighbor` | .6474±.1764 | .7705±.0170 | **+.1231±.1809** |
| Delta-wing resampled | Linear, latent 8 / `fmt_all` | .4274±.0274 | .8250±.0208 | **+.3976±.0269** |
| Delta-wing original LBM | Linear, latent 8 / `fmt_all` | .4619±.0513 | .8524±.0031 | **+.3905±.0538** |
| F-22 | MLP `[256,128]`, latent 16 / `fmt_all+kin4` | .6670±.0152 | .5958±.0202 | **−.0712±.0297** |
| Boeing 747 | Linear, latent 4 / `kin2` | .5801±.0129 | .8114±.0300 | **+.2313±.0366** |
| Smoke buoyancy | Linear, latent 16 / `fmt_real_imag_neighbor` | .5184±.1489 | .7163±.0055 | **+.1980±.1538** |
| **Dataset macro** | — | **.4617** | **.6308** | **+.1691** |
| **Family macro** | — | **.4760** | **.6373** | **+.1613** |

Development / outer / final 的 dataset-macro F1 增益依次为 `+.21956 / +.21557 / +.16912`。最终效应有所回落，但仍超过预注册目标。F-22 在三阶段均是负例；它没有被删除或换标签。

## Task3：监督式 whole-field IVD p95 二分类

`geometry+FMT` 表示 residual head 同时读取冻结 Raw backbone 的 geometry embedding 与 FMT；`FMT-only` 表示 residual head 只读取 FMT。`aux64` 是 FMT 映射后的 64 维 auxiliary vector。所有比较中的 Raw-PCA residual 与对应 FMT residual 具有相同 auxiliary width、residual route 和训练设置。

| Flow | 冻结 FMT residual 配方 | Raw-PCA F1 | FMT F1 | F1增益 | Raw-PCA AP | FMT AP | AP增益 |
|---|---|---:|---:|---:|---:|---:|---:|
| Channel observer | `fmt_all+kin4`, geometry+FMT, aux64, lr `3e-4` | .3058±.0248 | .7554±.0148 | **+.4495±.0311** | .2725±.0359 | .8394±.0164 | **+.5669±.0520** |
| Half-cylinder Re160 | `kin2`, FMT-only, aux64 | .7146±.0092 | .7335±.0159 | **+.0189±.0150** | .7882±.0155 | .8454±.0133 | **+.0572±.0061** |
| Half-cylinder Re640 | `kin2`, FMT-only, aux64 | .6908±.0057 | .8130±.0104 | **+.1222±.0147** | .7776±.0112 | .9037±.0114 | **+.1261±.0079** |
| Half-cylinder Re6400 | `kin2`, FMT-only, aux64 | .6143±.0098 | .7267±.0140 | **+.1124±.0137** | .6682±.0111 | .8188±.0104 | **+.1507±.0120** |
| Tangaroa | `kin2`, geometry+FMT, aux64, alpha `1` | .7740±.0062 | .8289±.0067 | **+.0548±.0078** | .8522±.0063 | .9158±.0070 | **+.0636±.0015** |
| Delta-wing resampled | `fmt_all+kin4`, FMT-only, aux64 | .8423±.0090 | .8915±.0098 | **+.0492±.0117** | .9269±.0047 | .9668±.0027 | **+.0400±.0058** |
| Delta-wing original LBM | `fmt_all+kin4`, FMT-only, aux64 | .8313±.0043 | .8834±.0098 | **+.0521±.0106** | .9228±.0043 | .9642±.0020 | **+.0414±.0038** |
| F-22 | `kin6`, geometry+FMT, aux64, lr `3e-4` | .8196±.0059 | .8410±.0088 | **+.0214±.0118** | .8813±.0126 | .9120±.0054 | **+.0307±.0081** |
| Boeing 747 | `fmt_all+kin2`, geometry+FMT, aux64, lr `3e-4` | .7692±.0166 | .8519±.0148 | **+.0827±.0170** | .8632±.0133 | .9373±.0059 | **+.0741±.0118** |
| Smoke buoyancy | `kin2`, geometry+FMT, aux64, alpha `1` | .7785±.0215 | .8152±.0136 | **+.0367±.0323** | .8813±.0118 | .9202±.0103 | **+.0389±.0077** |
| **Dataset macro** | — | **.7141** | **.8141** | **+.1000** | **.7834** | **.9024** | **+.1190** |
| **Family macro gain** | — | — | — | **+.1115** | — | — | **+.1323** |

Development / outer / final 的 dataset-macro F1 增益依次为 `+.11963 / +.14027 / +.10000`，Average Precision 增益为 `+.12805 / +.14623 / +.11896`。最终 10/10 条目方向一致，但主对照的 F1 增益没有达到 `+.15`。相对标准 Raw / Raw-wide 的诊断性 dataset-macro F1、Average Precision 增益为 `+.15828 / +.19479`；不能用这组较弱对照替代预注册的 Raw-PCA 主对照。

## 机器结果与复核

- Ibex Task2 jobs：`50929941`（10×V100 array）和 `50929943`（CPU summary）。
- Ibex Task3 jobs：`50929944`（10×V100 array）和 `50929946`（CPU summary）。
- Task2 `summary.json` / `per_run.csv` SHA-256：`06c7de3e0d6eae2e62761c2394985d455601fec076fb0c453fa676a1fdbe8122` / `c06992f5d69edf3f5c28256402968c1aca3c9896e4433c2fb6041905d96c0c78`。
- Task3 `summary.json` / `per_run.csv` SHA-256：`44d370709fd9d1af71d9b5f90e7ad60fc36da459a022ee8f9a6ba2012c46c25c` / `8ebfc2fd26c94c767cce6086ebb9bd318cd876c52a5bda2dabeda8a4f49a74be`。
- 本地复核通过：Task2 100/100、Task3 200/200 唯一方法-数据集-seed记录；selection SHA-256 一致；关键指标全部有限；远端与下载文件哈希一致。
- 实现 commits：`f8f5675`、`7bdb511`。完整运行登记见 `docs/ibex_run_registry.md`。
