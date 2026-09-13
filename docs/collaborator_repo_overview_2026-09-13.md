# FMT 代码库简介（给合作者，2026-09-13）

本文是入门导读，只讲三件事：FMT 核心在哪里、流体数据怎么读进来、每个 Task 的代码在哪里。
任务定义、评测边界和禁止事项以 [AGENTS.md](../AGENTS.md) 与 [研究协议](research_tasks_and_protocol.md) 为准，
本文不重复也不覆盖它们。

## 1. 仓库是什么

FMT（Flowmap Tokenizer）是一个**无可训练参数**的编码器：把一条中心轨线加六条邻居轨线组成的“七线 primitive”
编码成一个特征向量。当前主线是三维流场，共十个数据条目：`channel`、`cylinder3d`（half-cylinder Re160）、
`halfcylinderRe640`、`halfcylinderRe6400`、`tangaroa`、`deltaWing_resampled`、`deltaWing_LBM`、`f22raptor`、
`boeing747`、`smokeBuoyancy`。

顶层目录：`FMT_Utils/` 编码器与任务模型，`FLowUtils/` 流场读取与轨线积分，`experiments/` 实验入口，
`config/` 实验配置，`ibex_bash/` 集群提交脚本，`docs/` 协议与结果，`tests/` 单元测试，`tools/` 仓库检查。
所有脚本从仓库根目录以 `python -m experiments.<模块名>` 运行；输出写到 `outputs/`，数据和模型不进库。
依赖见 `requirements_fmt.txt`（含 CUDA 相关项，尚未整理为跨平台安装包）。

## 2. FMT 核心位置

- **主实现**：[FMT_Utils/DFT_FMT_3D.py](../FMT_Utils/DFT_FMT_3D.py)，入口函数 `pathline_dft_features_3d`。
  输入张量形状 `[N, 7, L, C]`：N 个 primitive，7 条线按 `center, x+, x-, y+, y-, z+, z-` 排列，L 个时间采样点，
  C 为 `(x,y,z)` 或 `(x,y,z,t)`。输出每个 primitive 一个特征向量。
- **做法**：每个时刻取六个邻居相对中心的偏移矩阵 `D(t)`（6×3），用 Gram 矩阵 `G(t)=D(t)D(t)^T` 消掉刚体观察者的
  旋转和平移，再对这些标量时间序列做实数离散傅里叶变换（默认 6 个频率），另加手性项。因此特征对时变刚体变换客观。
- **特征块选择**：`fmt_feature_indices_3d(name)` 按名字取子块，配置里的 `fmt_all`、`fmt_real_neighbor`、
  `gram2`、`kin4` 等候选名就来自这里；缓存到特征矩阵的组装在 [FMT_Utils/Task12Data_3D.py](../FMT_Utils/Task12Data_3D.py)。
- **变体**（各有独立实验版本，不能与主实现混称）：`FMT_Utils/FMTAllV2_3D.py`（客观邻居几何再做时间傅里叶）、
  `FMT_Utils/objective_fmt_nTDO.py` 与 `objective_fmt_nTDO_v2.py`（只用点间距离标量的客观版本）、
  `FMT_Utils/Task6Recovery_3D.py::signed_fmt`（Task6 保留复系数方向的 token）、
  `FMT_Utils/Task5FeatureRecipes_3D.py`（Task5 可变尺度配方）。
- `FMT_Utils/FMT_encoder.py`、`DCT_FMT_encoder.py` 是 PyflowVis 时期的二维可学习编码器，只被 `experiments/FMT_Clustering.py`
  使用，不属于三维论文主线。

## 3. 流体数据如何访问

数据不在仓库里。配置文件同时写了 Windows 本地路径（`.../FLowVisAssets/flowData3D/`）和 Ibex 路径
（`/home/zhanx0o/DeepVortex/FLowDataFolder/`、`/ibex/user/zhanx0o/FLowDataFolder/`），例如
`config/mainExp_Task5_3D_1.1.yaml` 的 `datasets` 段；换机器只需改这些路径。

读取与预处理链路：

1. **原始场**是 NetCDF `.nc`，规范布局 `[T, Z, Y, X, 3]`；`channel` 是 JHTDB 的定常 `channel.vtk`，
   用时变 Killing 观察者生成等价的非定常时刻（`FMT_Utils/KillingObserver3D.py`、
   `experiments/Build_Channel_Killing_Cache.py`）。JHTDB 采样下载工具在 `FLowUtils/flowDatasetUtils/JHTDB_*.py`
   与 `experiments/Download_JHTDB_*.py`，说明见 `docs/JHTDB_*.md`。
2. **加载**：`FMT_Utils/FMT_3D_pipeline.py::load_vector_field_3d` 调用
   `FLowUtils/flowDatasetUtils/NetCDF_AmiraLoader.py::NetCDFLoader.load_vector_field3d`，自动识别分量变量名
   （u/v/w、x/y/z 等），返回 [FLowUtils/VectorField3d.py](../FLowUtils/VectorField3d.py) 的 `UnsteadyVectorField3D`
   （按时间取切片、物理坐标与网格坐标互转、任意点取速度）。大场用
   `FMT_Utils/NetCDF_window_3D.py::load_netcdf_window_3d(path, start_index, frame_count, max_spatial_dim=96)`
   只读一个时间窗并按步长降采样空间网格。
3. **播种与积分**：`FMT_3D_pipeline.generate_seeding_grid_3d` 生成内缩的规则种子网格，
   `integrate_cross_primitives_3d` 用 RK4 积分七线 primitive（底层是
   `FLowUtils/flowlineIntegral.py::compute_pathlines_3D_batch`）；Task5 的可变尺度版本在
   `FMT_Utils/MultiscalePathline_3D.py::integrate_multiscale_primitives_3d`。
4. **参考标签**：`FMT_3D_pipeline.compute_ivd_reference_3d` 与 `FLowUtils/ScalarField3d.py::compute_ivd_3D`
   计算 IVD（instantaneous vorticity deviation，瞬时涡量偏差）；论文中的涡/非涡二分类统一用 IVD 的 p95 阈值。
5. **缓存**：每个时间片的 primitive 与参考量存为 NPZ，位于 `outputs/<实验ID>/cache/`。
   Task1/2/3 的主实验直接读缓存而不重新积分，读取函数是 `Task12Data_3D.load_cache_records`，
   缓存目录写在各配置的 `sources` 或 `source_cache_root` 字段（例如 `outputs/Verify_Task2Universality_1.1/cache`）。

Cylinder 类数据的默认起始时间范围见 `config/cylinder_time_policy.json` 与 `FMT_Utils/CylinderTimePolicy.py`。

## 4. 各 Task 代码位置

| Task | 一句话定义 | 主入口脚本 | 配置 | 核心库模块 | 协议与结果 |
|---|---|---|---|---|---|
| Task1 | FMT 特征直接做涡/非涡 KMeans 二聚类 | `experiments/Run_Task1_3D_Uniform.py`（复用 `Run_Task1_3D_Main.py`） | `config/mainExp_Task1_3D_4.1_uniform.yaml` | `FMT_Utils/DFT_FMT_3D.py`、`Task12Data_3D.py`、`Task12Evaluation_3D.py` | `docs/mainExp_Task1_3D_2.1.md`、`docs/paper_tables_tasks_3d.md` |
| Task2 | 同一个冻结 VAE，比较 Raw 轨线输入与 FMT 输入的 latent 聚类 | `experiments/Run_UniformFMT_Confirmation_3D.py`；搜索 `Search_Task2_FMTVAE_3D.py` | `config/mainExp_Task2_3D_6.2_uniform_confirmation.yaml` | `FMT_Utils/VAE_3D.py`、`RawPathline_3D.py`；训练器 `experiments/Verify_HighReVAE.py` | `docs/mainExp_Task2_3D_5.2.md`、`docs/paper_tables_tasks_3d.md` |
| Task3 | 有监督涡区域二分类，比较不加 FMT 与加 FMT 残差分支的同结构网络 | `experiments/Run_UniformFMT_Confirmation_3D.py`；搜索 `Search_Task3_FMTResidual_3D.py`；标签 `Build_Task3_GlobalIVD_Labels.py` | `config/mainExp_Task3_3D_9.2_uniform_confirmation.yaml` | `FMT_Utils/PathlineClassifier_3D.py`（`PathlineFMTResidualClassifier3D`）；训练器 `experiments/Verify_Task3_FMTResidual.py` | `docs/mainExp_Task3_3D.md`、`docs/paper_tables_tasks_3d.md` |
| Task5 | Task3 的多尺度扩展：邻居距离、积分步长、步数可变，在未见尺度组合上二分类 | `experiments/Build_Task5_Multiscale_Cache.py` 建缓存，`Evaluate_Task5_Multiscale.py` 训练评估 | `config/mainExp_Task5_3D_1.1.yaml`、`mainExp_Task5_3D_1.1_evaluate.yaml` | `FMT_Utils/MultiscalePathline_3D.py`、`Task5FeatureRecipes_3D.py` | `docs/mainExp_Task5_3D_1.1.md` |
| Task6 | 单流场多尺度 primitive 的 VAE 几何重建：冻结 FMT token 进 VAE，解码整簇七条轨线 | `experiments/Task6_Reconstruction_3_1.py`；根目录 `Submit_Task6_*.py` 为 Ibex 提交 | `config/mainExp_Task6_Reconstruction_3.1.json` | `FMT_Utils/FlowMapData_3D.py`、`PrimitiveVAE_3D.py`、`Task6Recovery_3D.py` | `docs/Task6_reconstruction_protocol_3.1.md`；后续 4.1/5.1/PNN/MoE 见 `docs/Task6_*.md` |
| Task3+6 联合（进行中） | 共享门控网络同时做 Task3 分类与 Task6 重建 | `experiments/Task36_BalancedScale_1_2.py`；`Submit_Task36_BalancedScale_1_2.py` | `config/Verify_Task36_BalancedScale_1.2.json` | `FMT_Utils/Task36Balanced_3D.py`、`Task36ScaleData_3D.py`、`Task36Labels_3D.py` | `docs/Task36_balanced_scale_protocol_1.2.md` |

Task4（三维涡类型多分类）另有独立协议与代码（`FMT_Utils/Task4A_*`、`Task4B_*`），与 Task3 严格分开，见协议文档。
Task7/8 已暂停，历史定义见 `docs/Task678_flowmap_protocol_1.1.md`。

## 5. 结果与记录在哪里

- 论文主表：[paper_tables_tasks_3d.md](paper_tables_tasks_3d.md)。
- 每个实验的方法级结论：[experiment_log.md](experiment_log.md)；验证与负结果总表：[Verify_experiments.md](Verify_experiments.md)。
- 每个 Ibex 作业：[ibex_run_registry.md](ibex_run_registry.md)。
- 版本命名规则（`mainExp_`、`Verify_`、`Ablation_`、`Other_`）与“冻结基线、版本化新方法”的工作方式见 `AGENTS.md`。
- 已从工作目录移出的历史脚本可按 [repository_maintenance.md](repository_maintenance.md) 恢复。
