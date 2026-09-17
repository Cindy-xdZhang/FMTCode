# 六流场Task1–3：模式a与模式b

<!-- fmt-convolution-withdrawal-notice-20260917 -->
> **2026-09-17撤销通知：** 本文旧Raw卷积＋FMT/残差及二维FMT卷积方案已撤销，不能重跑或用作当前FMT主表、附录、“最佳方法”证据。历史数值仅供核对。无卷积的固定特征和全连接分支另行保留；以 [FMT禁止空间卷积协议](FMT_no_spatial_convolution_protocol_1.1.md) 为准。
<!-- fmt-convolution-withdrawal-notice-20260917-end -->

版本 `Other_Task123_PaperTriptychs_1.2`。固定流场顺序：Half-cylinder Re160、Re640、Re6400、
Tangaroa、Boeing747、Delta-wing原始LBM。每流场三任务，各有论文版和PPT版。
2026-09-06全部完成：18组三联图、36张版式图片、126个格式文件；
[下载全部图片包（28.1MB）](../outputs/Other_Task123_PaperTriptychs_1.2/Task123_SixFlows_ModeAB_figures.zip)。

| 模式 | 用途 | 三栏内容 | 代码 |
|---|---|---|---|
| a | Task1 | 几何体＋IVD等值面＋轨线；FMT两类聚类；FMT相对IVD的误检/漏检 | [渲染器](../experiments/Visualize_Task123_PaperTriptychs_1_2.py)的`draw_background`及`mode_a`分支；结构继承旧Task1三联图 |
| b | Task2/3 | IVD参考；无FMT匹配对照；FMT | 同一渲染器的普通分支，继承1.1模式b |

IVD全称Instantaneous Vorticity Deviation（瞬时涡量偏差），本版显示面和评估标签都使用p95。
旧模式a显示面为p97；本版明确统一到p95，不改写旧图。
Task1两簇经原协议校准为涡区/非涡区，第二栏显示全部种子，红为涡区、蓝为非涡区。
误差栏红圆点为正确涡点、紫三角为误检、黄叉为漏检；正确非涡点省略显示但保留于全部指标。
模式b的Task3对照仍为同结构Raw-PCA residual；PCA即Principal Component Analysis（主成分分析）。

## 文件与版式

[论文图片目录](../outputs/Other_Task123_PaperTriptychs_1.2/paper/)：7.2×2.9英寸，最小字号7pt，PDF、SVG、400dpi PNG、600dpi TIFF。
[PPT图片目录](../outputs/Other_Task123_PaperTriptychs_1.2/slides/)：13.33×5.4英寸，最小字号13pt，PDF、SVG、220dpi PNG。
文件名 `{dataset}_{task}_triptych`；六流场ID分别为 `cylinder3d`、`halfcylinderRe640`、
`halfcylinderRe6400`、`tangaroa`、`boeing747`、`deltaWing_LBM`。
最终图片包及按任务排列的预览位于上述输出根目录。预览供浏览，论文使用单张正式PDF。

## 数据与几何边界

沿用Task1-4.1、Task2-6.2、Task3-9.2当前统一配方；随机种子7080/100/40，确认时间片序号0/8/8。
这是预先指定的单次空间示例，不从确认数据选择漂亮图片，不替代重复实验主表。
Task1与Task2/3不同原协议的确认时间范围不同，不声称跨任务为同一时刻。
Re160与Tangaroa直接复用1.1预测，其他四流场复放同一冻结配方。预测数组和模型不下载。

[显示资产构建器](../experiments/Build_Task123_DisplayAssets_1_2.py)从本地原始速度场重建12份参考资产，
逐点核对原缓存p95标签。远端渲染再逐项核对种子坐标、标签、时间、原始帧序号、采样步幅、载入形状和阈值，
通过才允许将本地场景与远端预测合并。没有预测插值曲面。

Task1显示轨线从同片种子中按IVD标签分层最大间距选取，沿用旧选点函数及随机种子7068，
目标70%涡区、其余背景。普通流场240条，Boeing/Delta-wing480条；RK4积分96步/192步，
步长为原始时间间隔四分之一。对原始时变速度做线性插值，首次越域即停止；若后续帧不足，
按可用帧数缩短并记录实际步数。这些只是说明轨线，不替换模型原始primitive。

Boeing与Delta-wing使用旧代码已核实的仿真几何及坐标变换。旧规范没有提供Re160/Re640/Re6400/Tangaroa
的独立、已核实几何，所以这些流场首栏只有原始场IVD面和轨线，没有依据零速度臆造障碍物。
几何简化沿用旧Boeing顶点分箱方法，未改变物理位置。为使实体轮廓可读，采用固定说明图层：
IVD面、半透明轨线、实体几何；不能将此图层顺序解释为精确遮挡测量。

## 图注草稿

**Mode a / Task1.** The first column shows the IVD-p95 reference surface and
time-coloured pathlines, with verified simulation geometry where available.
The second column shows the two FMT+KMeans clusters, calibrated as vortex
(red) and non-vortex (blue). The third column compares the vortex prediction
with the same IVD-p95 reference: red circles indicate true positives, purple
triangles false positives, and ochre crosses false negatives. Path colours
encode elapsed integration time; display trajectories do not change the model
inputs. All three panels share spatial bounds, physical aspect ratios and camera.
F1 is the harmonic mean of precision and recall for the displayed snapshot.

**Mode b / Tasks2–3.** Columns show the IVD-p95 reference, the matched comparison
without FMT, and the corresponding FMT method. Prediction markers represent
actual evaluated seeds. Red circles, purple triangles and ochre crosses denote
true positives, false positives and false negatives, respectively. True negatives
are omitted from the display but included in F1. Task2 uses the same frozen VAE
architecture and training recipe in both arms. Task3 compares matched Raw-PCA
and Raw+FMT residual branches. Snapshot scores are not aggregate main-table results.

## 质量检查

面板尺寸以1.5pt容差强制检查；最终PDF逐份检查字号、碰撞与裁剪，并逐栏查看图片。
源码静态预检19 PASS、2 WARN、0 FAIL；动态字号由最终PDF确认，单次示例不适用重复实验误差条。
数据审计在Ibex执行、报告留远端；本地仅保存已下载图片的哈希和PDF排版审计。
最终检查记录见输出根目录的`delivery_manifest.json`，作业记录见[Ibex登记表](ibex_run_registry.md)。

完成核查：远端独立审计退出码0；36份面板尺寸PASS，逐片指标重算、场景身份和临时模型清理检查通过。
本地36份最终PDF均0 FAIL、最小字号7pt/13pt。每份有1项z方向标跨白色背景边缘WARN，
逐栏目视确认文字完整且不遮挡数据后接受；原自动审计的REVIEW REQUIRED状态保持不改写。
已通过按任务排列的完整论文/PPT联系表逐栏查看全部36图，并单独查看飞机几何及代表图。
三张总览只供浏览；不得把总览縮小后作为具有原始7pt字号的单幅论文图。
