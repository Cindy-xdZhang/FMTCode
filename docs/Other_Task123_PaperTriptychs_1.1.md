# Re160 / Tangaroa Task1–3 效果图

<!-- fmt-convolution-withdrawal-notice-20260917 -->
> **2026-09-17撤销通知：** 本文旧Raw卷积＋FMT/残差及二维FMT卷积方案已撤销，不能重跑或用作当前FMT主表、附录、“最佳方法”证据。历史数值仅供核对。无卷积的固定特征和全连接分支另行保留；以 [FMT禁止空间卷积协议](FMT_no_spatial_convolution_protocol_1.1.md) 为准。
<!-- fmt-convolution-withdrawal-notice-20260917-end -->

2026-09-06完成。版本 `Other_Task123_PaperTriptychs_1.1`；绘图方法索引P13。
两流场各三任务，共6组科学内容；每组分别提供论文版与组会PPT版。

| 用途 | 输出 | 画幅与最小字号 | 文件格式 |
|---|---|---|---|
| 论文 | [paper](../outputs/Other_Task123_PaperTriptychs_1.1/paper/) | 7.2×2.9英寸；7pt | PDF、SVG、400dpi PNG、600dpi TIFF |
| 组会PPT | [slides](../outputs/Other_Task123_PaperTriptychs_1.1/slides/) | 13.33×5.4英寸；13pt | PDF、SVG、220dpi PNG |

文件名为 `{cylinder3d,tangaroa}_{task1,task2,task3}_triptych`。
`cylinder3d`对应Half-cylinder Re160。[全部图片ZIP](../outputs/Other_Task123_PaperTriptychs_1.1/Task123_Re160_Tangaroa_figures.zip)仅包含42个图片文件。

## 三栏含义与图注

左栏为Instantaneous Vorticity Deviation（IVD，瞬时涡量偏差）第95百分位参考面及涡区采样点；
中栏为不使用FMT的匹配对照；右栏为FMT方法。三栏共享种子点、坐标范围、物理比例与相机。
红圆点表示正确识别的涡点，紫三角表示误检，黄叉表示漏检。正确识别的非涡点不画出，但仍参加指标计算。
橙色参考面来自原始速度场；预测只画实际样本点，没有把插值点当成额外预测。
Task3对照具体为同结构Raw-PCA残差分支；PCA是Principal Component Analysis（主成分分析），用于线性降维。

可用英文图注：

> Qualitative vortex-identification examples in Half-cylinder Re160 and Tangaroa.
> Columns show the IVD-p95 reference, the matched comparison without FMT, and the
> corresponding FMT method. Red circles denote true positives, purple triangles
> false positives, and ochre crosses false negatives. True negatives are omitted
> from the display but included in the reported F1 score. The reference surface is
> reconstructed from the original velocity field; predictions are shown only at
> evaluated pathline-primitive seeds. Each triptych uses identical seeds, spatial
> bounds, physical aspect ratios and camera settings. F1 denotes the harmonic mean
> of precision and recall for the displayed snapshot, not the aggregate main-table
> result. Task3 compares a Raw-PCA residual branch against a Raw+FMT residual branch.

## 实验身份与复现

| 任务 | 当前统一配方 | 固定随机种子 | 固定确认时间片序号 |
|---|---|---|---|
| Task1 | mainExp_Task1_3D_4.1 | 7080 | 0 |
| Task2 | mainExp_Task2_3D_6.2_uniform_confirmation | 100 | 8 |
| Task3 | mainExp_Task3_3D_9.2_uniform_confirmation | 40 | 8 |

使用各自协议首个注册种子和首个确认时间片，未依图像或确认指标挑选。
Task1与Task2/3的原协议时间范围不同，因此不能声称三任务为同一时刻。
复放既有配置不保证与原训练逐位相同，不覆盖冻结主表。本次不据单张图新增方法优劣结论。

- [配置](../config/Other_Task123_PaperTriptychs_1.1.json)
- [预测导出](../experiments/Export_Task123_PaperTriptychs_1_1.py)
- [三联图渲染](../experiments/Visualize_Task123_PaperTriptychs_1_1.py)
- [独立数据审计](../experiments/Audit_Task123_PaperTriptychs_1_1.py)
- [Ibex渲染入口](../ibex_bash/other_task123_paper_triptychs_1p1_render.sh)

Ibex工作目录 `/home/zhanx0o/FMT_Uniform_3D_20260901`，部署脚本在 `visualization_1p1/`，
使用既有deepvortex环境；输出位于该目录下 `outputs/Other_Task123_PaperTriptychs_1.1/`。
用户授权仅下载图片，逐点预测、源数据CSV、逐图元数据和数据审计报告保留远端。
本地 `delivery_manifest.json`仅记录已收到图片的哈希、尺寸与本地质量检查。
需要改字号、相机或版式时，使用已有远端预测单独运行渲染入口，不必重训。

## 检查记录与边界

预测job51391980[0-5]完成；最终渲染job51392430完成。
独立数据审计在远端执行，退出码0：逐点指标重新计算、预测哈希、参考标签零错配、12份面板检查PASS、
42个图片存在及临时checkpoint清理断言均通过。复放与冻结主表的差异记录保留远端，不据此选图。

本地12份最终PDF最小字号分别为7pt/13pt，均无低于5pt文字；碰撞检查均0 FAIL、1 WARN。
每个WARN均是方向标z文字跨越白色绘图区背景边缘；已逐图查看，背景两侧均白色，文字完整，
不遮挡数据或其他文字，接受此有意放置。自动审计原状态仍保留REVIEW REQUIRED，未改成无警告PASS。
12张最终PNG均已逐栏与整图检查。首版默认三维刻度发生重叠，修订为共享方向标与数值范围后重检。

最终渲染源码静态预检19 PASS、2 WARN、0 FAIL：动态字号已由实际PDF测量确认；
不确定性警告为关键字检查触发，本图展示单次预先指定样本，没有重复实验均值，不适用误差条。
以上为当前尺寸下的检查；排入论文后若进一步缩小，须重新检查最终字号。

最终渲染源码SHA256：`0800496983707cfec6160bca9a9969f89e624500c2f42c850bb2d406d4b20665`。
代码基础commit：`b0fe6d274aacf5f79d333505c2d454ccdffc3dbc`，以上新增脚本与配置尚未提交。
