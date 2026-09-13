# 绘图方法、代码与成图索引

核查日期：2026-09-06。用途：Task1/2/3 论文效果图与组会 PPT，后续每次出图继续更新。
本页记录绘图实现与数据接口，不新增方法级实验结论；实验结论仍以
[实验记录](experiment_log.md) 为准。主表身份以 [论文结果表](paper_tables_tasks_3d.md)
及 [研究协议](research_tasks_and_protocol.md) 为准。

## 当前实验与图像来源

截至本次本地记录核查，现行主结果仍为 Task1 4.1、Task2 6.2、Task3 9.2。
2026-09-03 的强对照与组件消融复现、09-04 的噪声类型扫描属于补充实验，未替换主结果；
09-05 摘要勘误修正了部分解释，没有重算主表。09-06 最新提交属于 Task4-b，不用于本页三任务效果图。
以上为索引初建时的记录；随后已在 Ibex 复放当前配方并完成 P13 图片，见本页末尾。

下表数值来自本地各任务 `paper_table.csv` 的10个数据条目等权平均，本次独立重算；
与现有柱状图 `source_manifest.json` 中对应文件 SHA-256 全部一致。
F1 是精确率与召回率的调和平均；IVD（Instantaneous Vorticity Deviation，瞬时涡量偏差）
衡量局部涡量相对空间平均涡量的偏离。三个任务固定使用全流场 IVD 第95百分位标签。

| 任务 | 主实验版本 / 机器表 | 对照 F1 → FMT F1 | 对照身份与出图限制 |
|---|---|---|---|
| Task1 | `mainExp_Task1_3D_4.1`；[机器表](../outputs/mainExp_Task1_3D_4.1_ibex/paper_table.csv) | .451024 → .601384 | 统一 Raw-PCA2 对统一 `fmt_all+kin4 → PCA8`；PCA（Principal Component Analysis，主成分分析）用于线性降维。旧逐 family 配方图不能标成4.1 |
| Task2 | `mainExp_Task2_3D_6.2`；[机器表](../outputs/mainExp_Task2_3D_6.2_uniform_confirmation/confirmation/paper_table.csv) | .488444 → .584040 | 原始轨线对 FMT，均输入同一变分自编码器（Variational Autoencoder，VAE）；统一 `fmt_all+kin4`、隐藏层 `[512,256]`、latent 64 |
| Task3 | `mainExp_Task3_3D_9.2`；[机器表](../outputs/mainExp_Task3_3D_9.2_uniform_confirmation/confirmation/paper_table.csv) | .639844 → .858338 | 同结构 Raw-PCA residual 对统一 `aivd1w3_dft` 辅助残差分支；图中应明确对照身份 |

主实验的轨线组固定为7条线、每线32个输出点、48步积分、`dt_scale=.25`、
`offset_grid_scale=.5`。显示用长轨线、额外局部采样、裁剪和插值均应另记，不能改写这一输入协议。
截至本次扫描，上述三个主结果目录内均未找到 `.npz` 文件；这只说明这些目录中没有此格式，
不证明其他位置或远端不存在逐点预测。仅靠 F1 汇总表不能重建空间效果图。

## 方法—代码—特点总表

新增独立方法 **P14 客观性验证四联图**：[代码](../experiments/Visualize_FMT_Objectivity_Translation_1_1.py)，
[参考系数值模块](../FMT_Utils/TranslationObserver_3D.py)，[说明](Other_FMTObjectivityTranslation_1.1.md)。
一列四栏0/25/50/100%目标observer平移速度；积分速度得到位移，在pushforward后的场中积分轨线；
同一分类器逐档重新编码，标签着色到整条中心轨线，保存分类变化和轨线一致性误差。
版本`Other_FMTObjectivityTranslation_1.1`；代码/解析场检查完成，正式目标observer待配置；不替代三联图a/b。

“已读”指核查源码与相关元数据，不表示本次成功复跑。绘图后端沿用项目保存的 Python。
编号仅是本页检索号，不是实验版本。

用户于2026-09-06固定两种三联图模式，后续沿用：

| 模式 | 三栏顺序 | 默认任务 | 代码与特点 |
|---|---|---|---|
| **模式 a** | 几何体＋IVD等值面＋轨线 → FMT两类聚类 → FMT相对IVD参考的识别误差 | **Task1** | 旧P02/P01的流程结构；新[1.2渲染器](../experiments/Visualize_Task123_PaperTriptychs_1_2.py)接入当前统一配方。独立几何仅用已核实仿真资产；显示轨线按IVD分层选择，不按预测挑选。新版显示/评估均p95 |
| **模式 b** | IVD参考 → 不使用FMT的匹配对照 → FMT | **Task2、Task3** | P13的现有三联图；[1.2渲染器](../experiments/Visualize_Task123_PaperTriptychs_1_2.py)扩展六流场。保留正确识别、误检、漏检；真实点，不插值生成预测曲面 |

新增批次 `Other_Task123_PaperTriptychs_1.2`：Re160、Re640、Re6400、Tangaroa、Boeing747、
Delta-wing原始LBM；Task1模式a，Task2/3模式b；不覆盖1.1。显示资产构建代码：
[Build_Task123_DisplayAssets_1_2.py](../experiments/Build_Task123_DisplayAssets_1_2.py)。
**已完成**18组×论文/PPT两版式，126个格式文件与3张总览；[图片与复用说明](Other_Task123_PaperTriptychs_1.2.md)。

| 编号 / 绘图方法 | 代码入口与核心函数 | 特点与适用场合 | 已有输出 / 当前限制 |
|---|---|---|---|
| P01 Task1 三张独立场景图 | [Visualize_Task1_3D_PaperCandidates.py](../experiments/Visualize_Task1_3D_PaperCandidates.py)：`_dense_scene`、`_draw_ivd_pathline_layers`、`_draw_two_cluster_layers`、`_draw_prediction_layers` | 等值面与按时间着色的轨线、两簇、预测误差分别输出；保留标题/图例，适合讲解 Task1 流程，也提供几何体和相机公共函数 | `outputs/Task1_3D_paper_candidates_1.1` 至 `1.9`；读取旧2.1/2.2逐 family 选择。默认仅6个流场；不可误称覆盖全部10项 |
| P02 Task1 横向三联图 | [Visualize_Task1_3D_Horizontal.py](../experiments/Visualize_Task1_3D_Horizontal.py)：`_new_horizontal_figure`、`_render_horizontal_scene`；复用 P01 | 左：IVD p97+几何体+轨线；中：FMT+KMeans两簇；右：相对IVD p95的预测误差。三栏同物理范围、正交相机；去标题/图例/色条，留坐标；适合论文组合排版 | `outputs/Task1_3D_horizontal_clean_1.1` 与 `Task1_3D_horizontal_halfcylinder_1.1`；21×5英寸、默认360 dpi、PNG+JSON；仍是历史配方 |
| P03 Task2/3 全域三联图及普通近景 | [Visualize_Task23_3D_Horizontal.py](../experiments/Visualize_Task23_3D_Horizontal.py)：`_validate_render_payload`、`_confusion_masks`、`_closeup_bounds`、`run` | 左：IVD p95参考；中：无FMT对照；右：FMT。严格同一批种子点/标签/预测；可分开生成预测和本地绘图；`--view-mode full/closeup` | `outputs/Task2_3D_horizontal_main_3.3`、`Task3_3D_horizontal_main_3.2`；默认读取旧Task2 3.3/Task3 3.2，ordinal 4，seeds 9068/40；不是当前6.2/9.2 |
| P04 全域+近景六联诊断图 / 通用三联渲染核心 | [Visualize_Task3_Multiflow_Diagnostic.py](../experiments/Visualize_Task3_Multiflow_Diagnostic.py)：`_render_one`、`_spec_close_up`、`_scattered_probability_mesh`、`_draw_scalar_slices` | 标准为2×3；设置 `close_up_only` 可作1×3。支持误差点、预测阳性点、连续曲面、正交切片；提供面板几何检查、蓝色像素检查及逐图JSON | `outputs/Task3_multiflow_diagnostic_1.2`；普通流场旧3.2，F-22另用历史anchored配方。默认 `focus_basis=paired_disagreement` 使用双方预测分歧；不能声称所有诊断裁剪均只依赖IVD |
| P05 Task3 真实加密点近景 | [Visualize_Task3_Multiflow_Dense5x.py](../experiments/Visualize_Task3_Multiflow_Dense5x.py)：`build_dense_cache`、`predict_dense`、`load_frozen_dense_prediction`；绘图委托 P04 | 重新积分20,480个局部 Sobol 样本（低差异空间采样），不是复制原点；`--stage cache/predict/render/all` 分阶段；仅画预测为涡的真实点，省略背景点 | [4.5配置](../config/Other_Task3DenseVisualization_4.5.yaml)；`outputs/Other_Task3DenseVisualization_4.5/figures`，复用4.4预测；seed404、zoom5.2、marker_scale .01；不是当前Task3 9.2 |
| P06 Task2/3 连续曲面宏观近景 | [Visualize_Task23_Multiflow_DenseCloseup.py](../experiments/Visualize_Task23_Multiflow_DenseCloseup.py)：`_load_task2_artifact`、`_load_task3_payload`、`_decorate`；绘图委托 P04 | 同一渲染核心按配置切换点/曲面；2.8开启 `surface_only`，无散点。Task2插值二值KMeans决策，Task3插值模型概率；适合展示局部边界，必须注明插值 | [2.8配置](../config/Other_Task23DenseCloseup_2.8.yaml)；`outputs/Other_Task23DenseCloseup_2.8`；112³网格、16邻居、距离反比幂1.75、平滑σ .6体素；当前该目录只发现Task3的两幅PNG，不能假设Task2已完成 |
| P07 F-22 独立误差诊断 | [Visualize_Task3_F22Anchored.py](../experiments/Visualize_Task3_F22Anchored.py)：`_vortex_roi`、`_clipped_mesh_triangles`、`_draw_prediction`、`render` | 提供物理裁剪、参考与误差绘制；也被 P03/P04复用；适合解释特定流场错误 | `outputs/Task3_F22Anchored_visualization_1.2`；专用冻结配方与模型读取，不能替代统一主实验 |
| P08 早期监督分类器四联诊断 | [Visualize_Task3_FMTClassifier.py](../experiments/Visualize_Task3_FMTClassifier.py)：`_plot_positive_score`、`_plot_confusion`、`visualize` | 1×4展示IVD减局部阈值、二值参考、Raw错误、Raw+FMT错误；`--config`、`--ordinal`，适合查模型行为 | 直接读取seed20模型文件；默认ordinal9；使用局部阈值标签接口，不能直接作为当前全场p95主图 |
| P09 统一主表性能柱状图 | [Visualize_Task1235_PerformanceBars_1_1.py](../experiments/Visualize_Task1235_PerformanceBars_1_1.py)：`build_records`、`create_main_f1_figure`、`create_ap_figure`、`export_figure` | 直接读取当前机器表，校验macro，输出整理后的CSV与来源哈希；灰色对照/深蓝FMT；含各数据条目、均值和误差线；适合论文量化结果和组会汇报 | [现有目录](../outputs/Other_Task1235_PerformanceBars_1.1/)；SVG/PDF/400 dpi PNG/600 dpi TIFF；现有三份碰撞检查JSON均记录PASS。本次核验源文件哈希，未重新渲染或重审PDF |
| P10 历史逐流场增益图 | [Plot_Task3_Universality.py](../experiments/Plot_Task3_Universality.py)：`plot_gains` | 从summary CSV画F1、Average Precision（平均精确率，概括精确率—召回率排序表现）增益；默认最低目标增益 .02 | 默认读取旧Task3 universality输出；阈值线是旧诊断目标，不是新主实验成功标准 |
| P11 Task5 横向三联图，可复用排版 | [Visualize_Task5_3D_Horizontal.py](../experiments/Visualize_Task5_3D_Horizontal.py) | 复用 P03 的布局与参考重建；固定三栏，支持 `--render-only` | `outputs/Task5_3D_horizontal_main_1.1`；本轮只登记，不纳入Task1/2/3结果 |
| P12 全流场参考面重建，非独立成图入口 | [VisualizationIVDReference_3D.py](../experiments/VisualizationIVDReference_3D.py)：`reconstruct_ivd_reference_surface` | 从原始场重算全域IVD、阈值及种子标签，检查source time/index、标签一致性与缓存身份；供P03/P04调用 | `outputs/IVD_reference_surface_cache`及各图目录内缓存。近景若启用 `render_reference_surface_from_labels`，实际显示面仍可能是标签插值面，不能仅凭有该审计就称其为原始连续IVD面 |

Task4 的体素图、四分类图另有 `Visualize_Task4_VoxelGT_3D.py`、
`Visualize_Task4B_Supervised_1_2.py`、`Visualize_Task4B_ChannelToTBL_2_3.py`、
`Visualize_Task4A_GeometricCurlKMeans_2_2.py`。本次仅定位，未做完整实现审阅，不能复用它们的类别含义到Task3。

## 三联图调用关系与数据要求

```text
Task1 Horizontal → Task1 PaperCandidates → 场景/轨线/几何体/簇/误差
Task23 Horizontal → IVD reference + Task1公共相机 + F22裁剪辅助
Task3 Dense5x ────────────┐
Task23 DenseCloseup ──────┴→ Multiflow Diagnostic._render_one
                              → IVD reference + F22几何裁剪
```

P03的输入必须一一对应：`seeds[N,3]`、`reference[N]`、两组 `predictions[N]`、
`bounds[2,3]` 和经过核查的IVD参考面。点密度倍数必须为1；需加密时重新积分并预测。
P04另需具名预测字典、逐方法指标、实验身份、原始时间片信息；曲面模式需要模型概率或
明确注明来源的二值决策场，并使用已冻结的阈值。不能用插值点冒充额外独立观测。

Task1显示面默认p97，评估仍为p95；展示轨线默认96步，模型输入仍为48步。
默认21³加密种子、240条显示轨线，其中70%来自涡区分层抽样；部分几何流场显示轨线数和步数乘2。
这些参数属于可视化采样，图注应说明，不能把显示占比解释为全场涡占比。

P03的普通近景仅由IVD标签选择。P04默认可使用双方错误分歧；P05/P06当前配置改为
`ivd_positive`。具体图以实际配置、`close_up_basis` 和逐图JSON为准，不能从文件顶部简介推断。
全部局部图的指标仍可能覆盖20,480个原局部样本，而画面只显示更小裁剪；图注应分别列出
`sample_count` 和 `close_up_visible_seed_count`。裁剪不能用于选择模型、阈值或主结论。

## 已有图片的核查与选用

| 代表图 | 本次查看结果 | 后续使用建议 |
|---|---|---|
| [Task1 Tangaroa横排](../outputs/Task1_3D_horizontal_clean_1.1/tangaroa_task1_horizontal_clean.png) | 三栏呈现轨线、红蓝簇、误差；保留物理坐标，无标题图例 | 可复用布局；论文需补齐图注和图例，标清历史配方或接入新预测 |
| [Task2 Boeing全域](../outputs/Task2_3D_horizontal_main_3.3/boeing747_task2_horizontal_clean.png) | 有蓝色背景点阵，与当前源码针对背景点的说明存在历史差异 | 图片实际外观不能由当前源码推断；需要新输出目录和完整来源记录 |
| [Task3 Re160连续曲面2.8](../outputs/Other_Task23DenseCloseup_2.8/task3_figures/cylinder3d_task3_closeup.png) | 参考面填满大部分画面，对照近乎空白，FMT为浅色大面；逐图JSON显示裁剪内仅20个原始点、指标仍来自20,480点 | 作为历史诊断保留；不直接作为代表性论文效果图。FMT区域 `all_positive` 且 `crop_boundary_closed=true`，裁剪边界封口不能误解为真实涡边界 |
| [Task3 Re160真实点4.5](../outputs/Other_Task3DenseVisualization_4.5/figures/cylinder3d_task3_closeup.png) | 对照无阳性点，FMT密集阳性点；局部范围极小。逐图JSON记录对照全负/FMT全正 | 这幅图展示局部分类退化，不能仅以红点多说明识别更准确；需保留假阳性/假阴性信息 |

本次只查看以上代表图，未逐张审核所有历史输出。记录中的 `predicted_positive_points=true`
在surface-only模式下不一定表示实际画点；应同时检查 `surface_only` 和
`scatter_collection_count`（2.8代表图为0）。这些是旧元数据字段语义问题，不修改冻结结果。

## 快速查找与复用

索引初建覆盖162个输出目录、1,438个PNG文件；加入1.1、1.2后为164个目录、1,489个PNG。完整成图目录清单见 [绘图产物CSV](plotting_artifacts_inventory.csv)：记录目录、
实际PNG/PDF/SVG/TIFF数量、元数据位置和配置是否仍在工作目录。目录存在不代表所有流场均完成。
归档配置可按 [仓库维护说明](repository_maintenance.md) 的清单在本地ZIP中定位；不必恢复全部历史文件。

以下仅为已读源码的命令接口，**本次未复跑**；P03会使用固定旧配置和固定输出目录，
P05/P06按配置中的输出目录写入。正式批量出新图前，应先设新可视化版本与新输出目录。

```powershell
# Existing Task2/3 prediction artifacts only; historical protocol.
python experiments/Visualize_Task23_3D_Horizontal.py --tasks task2 task3 --datasets tangaroa --render-only --view-mode full

# Real dense points; uses the config's frozen predictions and output directory.
$env:FMT_TASK3_VIS_CONFIG = 'config/Other_Task3DenseVisualization_4.5.yaml'
python experiments/Visualize_Task3_Multiflow_Dense5x.py --stage render --datasets tangaroa

# Continuous surfaces; do not use --recompute-task2 for a rendering-only revision.
$env:FMT_TASK23_DENSE_VIS_CONFIG = 'config/Other_Task23DenseCloseup_2.8.yaml'
python experiments/Visualize_Task23_Multiflow_DenseCloseup.py --tasks task3 --datasets tangaroa

# Current quantitative main tables; explicitly choose a new output directory.
python experiments/Visualize_Task1235_PerformanceBars_1_1.py --output-dir outputs/Other_Task1235_PerformanceBars_1.2
```

P06虽然未指定重新计算，Task2预测缓存缺失时仍可能进入训练；P03 `--render-only` 才明确要求
已有预测。不能把所有可视化入口都当作纯渲染程序。
多个旧脚本使用同目录裸模块导入，上面保留直接脚本调用方式；改成模块运行前需核查导入路径。
系统 `C:/Program Files/Python311/python.exe` 本次检查缺少 NumPy、Matplotlib、PyTorch、SciPy、
scikit-learn、netCDF4、Pillow、PyYAML，未检查所有其他Python环境，不能声称当前命令可直接运行。
后续应定位既有研究环境或适用运行时后再渲染；纯读图和本次CSV索引不依赖这些库。

## 后续每次出图应登记

先选本页方法与实际实验版本，再记录数据条目、时间片、随机种子、点数、模型/特征配置及源文件哈希。
三栏保持同数据、同范围、同相机；解释误差所需的假阳性、假阴性不可因追求简洁而被隐瞒。
主表对应图使用当前统一配方；旧逐family图可用于明确标注的历史进度/补充说明。

新图应导出论文PDF/SVG与PPT用PNG，并保存源数据、配置、逐图JSON。论文与PPT采用各自字号和画幅，
不能简单把21×5英寸整图缩到论文栏宽后宣称字体可读。新多面板输出需记录最终绘图区尺寸检查，
最终PDF检查文本、碰撞和裁剪，并逐栏看实际成图；旧图仅有JSON或旧PASS不等于新的输出已通过。
索引初建未改绘科学图片；随后新增P13，完成状态如下。

| 日期 | 绘图版本 | 方法编号 / 代码 | 数据与实验身份 | 特点/改动 | 输出与检查 |
|---|---|---|---|---|---|
| 2026-09-06 | 索引初建，无新实验 | P01–P12 | 本地主表、源码与代表图元数据 | 区分历史空间图、真实点、插值曲面与当前主表；查看4张代表图 | 本页及产物CSV；主表重算与来源哈希核对；未渲染 |

## P13 当前统一配方三联图：模式 b（1.1已完成）

| 绘图方法 | 代码 | 特点与状态 |
|---|---|---|
| 当前主实验：参考—无FMT对照—FMT | [渲染代码](../experiments/Visualize_Task123_PaperTriptychs_1_1.py)；[预测导出](../experiments/Export_Task123_PaperTriptychs_1_1.py)；[独立数据审计](../experiments/Audit_Task123_PaperTriptychs_1_1.py) | `Other_Task123_PaperTriptychs_1.1`，Re160/Tangaroa；原始速度场重建p95参考面，真实预测点展示正确/误检/漏检；论文7.2×2.9英寸、7pt，PPT13.33×5.4英寸、13pt；完成6组×2版式，共42个图片文件 |

配置 `config/Other_Task123_PaperTriptychs_1.1.json`；完整图契约
`outputs/Other_Task123_PaperTriptychs_1.1/figure_contract.json`。使用当前统一配方；
Task1首个确认ordinal0/seed7080，Task2 ordinal8/seed100，Task3 ordinal8/seed40；
不选图像最有利时间片。三任务预先固定的确认population并不相同，不声称跨任务同一物理时刻。
2026-09-06 Ibex 51391980[0-5]预测复放完成，51392430最终渲染完成。
用户授权仅下载图片，预测数组和远端审计报告留在Ibex；本地收到PNG/PDF/SVG/TIFF。
12份PDF面板尺寸检查通过，碰撞检查均0 FAIL；每份方向标z跨白色背景边界的1项WARN已逐图目视接受。
最小字号论文7pt、PPT13pt；12张最终成图全部逐栏查看。
输出：[论文版](../outputs/Other_Task123_PaperTriptychs_1.1/paper/)、[PPT版](../outputs/Other_Task123_PaperTriptychs_1.1/slides/)；
[使用说明与图注](Other_Task123_PaperTriptychs_1.1.md)。复用时只改展示参数可单独运行渲染脚本；
新增任务/流场必须另建版本及配置，不能拿现有图片指标替代主表。
本机已定位有Matplotlib的tradingagents Python3.13，额外科学计算/审计依赖只安装在项目
`tmp/task123_plotdeps/`，不改其他项目环境。前文“系统Python缺库”检查结果仍成立。

## Cylinder后续出图时间默认规则（2026-09-06）

所有cylinder/half-cylinder新图和新版本实验忽略原始模拟前50%，且起始t>=7；当前Re160/Re640/Re6400原始[0,15]对应下限7.5。Re640文件已裁到[7.5,15]，不重复截半。旧版示例的早期时间不再作为新图默认值；冻结结果仍保留。规则：`config/cylinder_time_policy.json`；选择器：`FMT_Utils/CylinderTimePolicy.py`。

## P15 全局平均平移客观性四联图（1.2）

| 绘图方法 | 代码 | 特点 |
|---|---|---|
| 全局平均速度 observer | [四联图](../experiments/Visualize_FMT_Objectivity_Translation_1_2.py)、[全局均值](../FMT_Utils/GlobalMeanObserver_3D.py)、[说明](Other_FMTObjectivityTranslation_1.2.md) | 完整源网格每时刻体积平均速度；速度积分为参考系位移；0/25/50/100%四档；同物质轨线、固定分类器、整条中心线分类着色；记录标签变化及固定IVD参考分数；全部轨线保留，非涡线低透明度避免遮挡。 |

## P16 Cylinder后半程客观性四联图（1.3，当前默认）

| 绘图方法 | 代码 | 特点 |
|---|---|---|
| 后半程全局平均速度observer | [四联图](../experiments/Visualize_FMT_Objectivity_Translation_1_3.py)、[时间选择](../FMT_Utils/CylinderTimePolicy.py)、[说明](Other_FMTObjectivityTranslation_1.3.md) | 继承P15，增加统一时间下限；只按元数据选首个合规缓存。Re160/Re640/Re6400初始t=10.5/9.5/10.5，图标题直接标出时间；旧1.2早期图保留但不再默认。 |

## P17 两倍显示时长、两倍数量的客观性四联图（1.4）

| 绘图方法 | 代码 | 特点 |
|---|---|---|
| 长轨线密集显示 | [四联图](../experiments/Visualize_FMT_Objectivity_Translation_1_4.py)、[完整轨线种子筛选](../FMT_Utils/DenseObserverPaths_3D.py)、[说明](Other_FMTObjectivityTranslation_1.4.md) | 96步/约2.4时长，数量7220/7154/7242；分类仍用前48步原32点。原完整种子优先、固定Sobol序列补足；不复制、不拉伸、不按标签选线；四档同种子、真实observed积分，记录越域排除。 |

## P18 七档平均平移速度客观性图（1.5）

| 绘图方法 | 代码 | 特点 |
|---|---|---|
| 纵向七档observer对比 | [七联图](../experiments/Visualize_FMT_Objectivity_Translation_1_5.py)、[七档计算](../FMT_Utils/SevenTranslationObservers_3D.py)、[说明](Other_FMTObjectivityTranslation_1.5.md) | 0至100%按1/6等间距改变observer速度，积分速度得到位移；复用1.4同一批7220/7154/7242条长轨线种子和原分类窗口；各档独立积分及分类，检查0/50/100%复现；保持各栏尺寸并纵向扩展，统一相机，禁用方形axes裁剪轨线。 |

## P19 相机worldline显式积分的observed七联图（2.1）

| 绘图方法 | 代码 | 特点 |
|---|---|---|
| 无bounding box的observed轨线 | [七联图](../experiments/Verify_FMT_Observed_Pathline_2_1.py)、[相机及场变换](../FMT_Utils/ObservedFieldWorldline_3D.py)、[说明](Verify_FMTObservedPathline_2.1.md) | 三步显式分开；数值积分相机速度、逆位置查询原场，再积分observed field；七档共同物质种子/尺度；不画原域盒或取景盒；精确变换和错误relative-field对照；分别诊断中心和邻居特征变化。 |

## P20 旧/新 FMT 的逐流场成对性能图

| 绘图方法 | 代码 | 特点 |
|---|---|---|
| 三任务成对 F1 对比 | [绘图](../experiments/Plot_FMTAllV2_Comparison.py)、[独立审计](../experiments/Audit_FMTAllV2_3D.py)、[汇总](../experiments/Summarize_FMTAllV2_3D.py) | 三栏分别表示 Task1、Task2、Task5；全部10条目、每条目5个成对种子；旧/新版均值连接，显示样本标准差及每次运行，不过滤下降条目。先要求完整审计 PASS 再绘制；固定0–1横轴，统一尺寸；导出可编辑PDF/SVG及600dpi PNG、源CSV和图注。 |


## P21 Task3标量移植到Task1/Task2的配对性能变化图

| 绘图方法 | 代码 | 特点 |
|---|---|---|
| 两任务配对F1变化、附加特征控制 | [绘图](../experiments/Plot_AIVDTransfer_Comparison.py)、[指标审计](../experiments/Audit_AIVDTransfer_3D.py)、[汇总](../experiments/Summarize_AIVDTransfer_3D.py) | 两栏分别为直接聚类和VAE后聚类；青色一维aivd1w3_dft、棕色加kin4，均相对重训旧配方；绘制5种子配对差的均值±样本标准差，末行按种子先求10条目平均。使用获准下载的聚合表，不伪造单次散点；100分片/350指标的完整预测审计在Ibex完成，本地仅核验聚合一致性。全部Cylinder t≥7.5；PDF/SVG/600dpiPNG及源CSV，面板尺寸、文字和碰撞检查。 |

## P22 aivd1w3_dft 七档平移 observer 验证图

| 绘图方法 | 代码 | 特点 |
|---|---|---|
| 固定标量分类器的七联图 | [计算与校验](../experiments/Verify_AIVDTranslationObservers_3D.py)、[绘图](../experiments/Plot_AIVDTranslationObservers_3D.py)、[协议](Verify_AIVDTranslationObservers_1.1.md) | 原样一维 aivd1w3_dft，晚期 Task1/seed7080 的固定 StandardScaler+KMeans；复用2.1七组独立 observed-field 积分轨迹，保留7220个共同物质 primitive；每档独立特征与预测，标出 changed labels，附特征误差和float64/精确变换诊断。七档速度、统一取景、无bounding box、96步显示/48步分类；仅验证平移，不以零changed labels证明一般旋转客观性。 |

## P23 七联图的极简相机与平移速度标注

| 绘图方法 | 代码 | 特点 |
|---|---|---|
| 右上角两笔矢量相机、精确速度比例 | [绘图与相机函数](../experiments/Plot_AIVDTranslationObservers_CameraLabels_3D.py)、[复用原轨线与标签](../experiments/Render_AIVDTranslationCameraLabels_3D.py)、[配置](../config/Verify_AIVDTranslationObservers_1.2.json) | P22注释层修订；相机为PathPatch轮廓+Ellipse镜头，paper大小16×11pt，黑线白底，可编辑PDF/SVG；draw_camera_velocity固定七档0、1/6、1/3、1/2、2/3、5/6、1乘以时变全局平均流速，不用常数近似或猜测单位。每格右上留白，原取景/轨线/分类不变，两版PNG新增注释区域外逐像素一致。原1.1图保留，输出1.2，字体/尺寸/碰撞/逐格审查完成。 |


## P24 相机旁的平移速度矢量箭头

| 绘图方法 | 代码 | 特点 |
|---|---|---|
| 相机、矢量公式、同比例速度箭头 | [绘图](../experiments/Plot_AIVDTranslationObservers_VelocityArrows_3D.py)、[原数据复用](../experiments/Render_AIVDTranslationVelocityArrows_3D.py)、[像素与箭头审计](../experiments/Audit_AIVDVelocityArrows_3D.py)、[配置](../config/Verify_AIVDTranslationObservers_1.3.json) | P23的注释修订：FancyArrowPatch绘制6个黑箭头、Ellipse绘制零速圆点；公式用vec矢量符号。初始时刻的真实observer速度经相同3D正交投影矩阵确定方向；最大60/90pt，其余按alpha缩放。图注明确箭头时刻与共享比例尺。无原始数据重算，两版注释区外像素变化0；PDF/SVG可编辑，完整QA通过。 |


## 2026-09-10 — 完整FMT直连网络训练曲线

| 绘图方法 | 代码 | 数据接口 | 特点与使用边界 |
|---|---|---|---|
| Task678小样本拟合九宫格 | `experiments/Plot_Task678_DirectFMTFit_1_1.py` | `Verify_Task678_DirectFMTFit_1.1` 的已完成memorize16条件、Task6/Task7原始learning_curve；快照JSON→CSV | 全九流场、不平滑；共享对数误差轴；实线/虚线区分任务，终点标记实际停止步数；0.001目标线；只显示训练误差，不能标为测试性能。183×157 mm，PDF/SVG及600 dpi PNG，附源数据和排版/字体/碰撞检查。独立于三联图模式a/b。 |


## 2026-09-10 — 密集流映射查询、方向保留编码与固定测试轨迹

完整方法—代码—接口—特点表见 [Task6/7/8绘图索引](Task678_plotting_methods_1.1.md)。新增三任务共同数据性能图、九流场三种编码的训练曲线，以及每流场3×4固定primitive真实轨迹图；全部要求先完成独立预测审计。新图不修改模式a/b定义。
