# 数据集 v2 样本查看器 2.1（分析工作台 1.6 第四页）

2026-09-19 用户要求在 hairpin 分类可视化页面里看到 v2 训练集/测试集的样本点、用滑块控制可见数量，并能按归属实例查看每个实例周围的样本。实现为工作台的第四页“数据集 v2 样本”，入口 `http://127.0.0.1:8767/Other_FMT_AnalysisWorkbench_1.6/index.html#dataset`（1.5 入口自动跳转 1.6，前三页逐文件继承 1.5）。

## 内容

- 数据：`mainExp_Task4C_FixedDataset_2.1` r3（`data_audit.json` SHA-256 e5aaa4fb…）。“测试折 0”= 折 0 全部样本（Channel 40,670、TBL 41,231）；“训练折 1–4”= 其余全部样本（Channel 148,746、TBL 157,182，含 FMT 2.1 的 10% 验证子集，点选详情会标出）。每个样本点一束三条曲线（短/中/长），物理坐标。
- 着色：默认“样本标签（无模型）”，TP/TN 即 hairpin/non-hairpin；可切换到 FMT 2.1 `p35_h0_fps6 · seed96611` 的测试折预测（行级预测按样本取三条长度的平均概率），训练折没有预测，切到训练折时自动回到标签着色。
- 区域：“真实标注区域”下拉只列出当前集合里有样本的实例，并显示划归该实例的样本数与 hairpin 数；勾选“区域按归属实例筛选”（默认）时显示划归该实例的全部样本（含按包围盒、候选分量、最近实例划入的负样本），视野覆盖 GT 曲面与这些样本；取消勾选则回到模板原来的“GT 包围盒附近”筛选。图上方面板给出该实例按归属方式与标签的分解，选 FMT 模型时另给正确识别/漏检/误报数。
- 数量：Hairpin / Non-hairpin 两个滑块（及数字框、“全部”按钮）控制可见样本数，默认各 100；几何按 512 样本一块按需加载。
- 其他：candidate head 区域叠加（与 1.2 相同的曲面文件）、点选详情（折、归属实例与方式、候选分量、中心 λ₂/oyf、FMT 概率）。

## 构建

- `experiments/Build_Task4C_DatasetViewer_2_1.py`（默认读本机 OneDrive 原始场核对哈希、生成 GT 曲面与几何分块；`--fmt-result` 指向已完成的 FMT 2.1 结果目录）→ `outputs/Other_Task4C_DatasetViewer_2.1/viewer/`（index.html 67 MB，几何分块共约 720 MB）。控件脚本 `experiments/templates/task4c_dataset_viewer_2_1.js`。
- `experiments/Build_FMT_AnalysisWorkbench_1_6.py --update-entry` → `outputs/Other_FMT_AnalysisWorkbench_1.6/`，1.5 入口改为跳转，原页面存为 `index_1.5.html`。
- 浏览器检查（2026-09-19）：测试折默认显示 200 样本；选实例 2 显示其 4,990 个归属样本的分解（标注单元内 1,029、包围盒内 3,017、最近 944）；切 FMT 模型后面板显示正确识别 171 / 漏检 858 / 误报 318、行级测试 F1 0.2254；切换到训练折时实例列表更新为 55 项、区域回到全部；滑块改变可见数生效；candidate head 叠加正常；控制台无报错。
