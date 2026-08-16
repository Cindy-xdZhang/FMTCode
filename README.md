# FMT — Training-free Objective Flowmap Tokenizer（重启仓库）

本仓库是 FMT 研究（对 pathline cross primitive 做无参数几何编码）从 PyflowVis 拆出的重启工作区，只承载三个任务：

- **Task1**：FMT encoder + KMeans 聚类区分 2D 非定常流的涡/非涡区域（入口 `FMT_Clustering.py`）。
- **Task2**：FMT 插入无监督 VAE 提升特征质量（**尚无实现**，PyflowVis 中也从不存在，需新写）。
- **Task3**：FMT 用于 3D 涡分类（spanwise / streamwise / hairpin；3D encoder 设计未定，暂不动）。

## 必读文档

| 文档 | 内容 |
|---|---|
| [docs/Table1_pyflowvis_review.md](docs/Table1_pyflowvis_review.md) | 表1：PyflowVis 六条研究线全梳理、FMT encoder 家族谱系、复制清单与修改记录 |
| [docs/first_principles_analysis.md](docs/first_principles_analysis.md) | 第一性原理问题分析（客观性逐环节审计、BN 隐患、DCT 负频率 bug、评测缺失、采样偏差等 P0–P9） |
| [docs/experiment_log.md](docs/experiment_log.md) | 实验版本表（唯一结论载体）+ 定量协议 |
| docs/from_pyflowvis/ | 原仓库中仅有的两篇相关文档（Vatistas 数据数学、GCD 多涡型命名规划） |

## 代码来源与修改声明

全部代码复制自 `C:\Users\xingdi\sources\PyflowVis`（main @ 3040ace，2026-07-23）。除以下两处外未做任何修改：
1. `FMT_Clustering.py`：`generate_FLowMap_SLICE` → `generate_Flowmap_SLICE`（上游改名导致的 ImportError，2 行）；
2. `requirements_fmt.txt`：UTF-16LE → UTF-8（原文件 pip 无法读取）。

已知仍处于"带病"状态、使用前需修复的文件见 first_principles_analysis.md 的 P7 表（如 `vortexExtraction_utiles.py` 无 import 语句、`pnn/models/point_nn.py` 硬编码 `.cuda()` 等）。

## 目录结构

```
FMT_Clustering.py            Task1 入口（已修复可 import）
FMT_Utils/                   encoder 家族（FMT_encoder / DCT_FMT / model_zoo）+ primitive 生成与 FTLE 工具
FLowUtils/                   2D/3D 场结构、pathline 积分（CUDA+CPU）、涡判据、Killing observer、NetCDF/Amira IO
DeepUtils/utils/             EasyConfig 等配置工具
pnn/                         Point-NN 改造版（EncNPNew）+ pathline 可视化（multi_points_vis_fast）
assets/cuda_kernal/          PathlineIntegration2D.cu（flowlineIntegral 按相对路径加载 → 必须从仓库根目录运行）
config/                      PathlineFMTclustering.yaml（Task1）、FittingVatistas / VatistasDataset（合成 GT）
FittingVatistasParam.py      Vatistas 参数拟合（定量评测的合成标签来源）
VatistasFlowDatasetGenerator.py
docs/                        表1、问题分析、实验记录
```

## 运行 Task1 基线（mainExp_1.1，待跑）

```bash
python FMT_Clustering.py
```

前置：`requirements_fmt.txt` 环境；数据路径在 `config/PathlineFMTclustering.yaml` 的 `dataset.dat_dir`（当前指向 OneDrive 的 flowData2d；beads2d / doublegyre2d / rfc2d 无数据文件时会走解析场生成）。CUDA 不可用时积分自动回落 CPU（慢几个数量级）。注意：跑任何实验前先读 `docs/experiment_log.md` 的协议。
