# Task4-c：分类查看器中的 candidate head 区域 1.2

2026-09-18 用户要求在 hairpin 分类可视化（分析工作台 1.5 第三页，即 `outputs/Other_Task4C_GTHeadCoverage_1.1/viewer`）中尝试显示 candidate head 区域：中心必须满足 λ₂ 涡判据（λ₂ < 阈值；Channel −13.395，TBL −0.0272）且展向涡量脉动 oyf > 0。这一条件与 `Task4C_fixed_dataset_rules_v1.md` 的 step1 播种筛选相同。本版本只增加显示数据，不改标签、预测、采样束或任何 F1。

## 区域定义与提取

- 候选区域 = {x : λ₂(x) < 阈值 且 oyf(x) > 0}，λ₂ 与 oyf 取原始 VTK 的 PointData，在点上按三线性插值判断（与采样代码 `interpolate_scalar` 一致），不是“单元八顶点全部满足”的旧候选单元定义。
- 边界曲面 = f(x) = max(λ₂(x) − 阈值, −oyf(x)) 的零等值面。f < 0 恰好等价于两条件同时成立（脚本内有 `np.array_equal` 断言）。用 scikit-image marching cubes（Lewiner，去除零面积三角形）在原生网格索引空间提取，再按各轴坐标线性映射回物理坐标；marching cubes 在网格边上线性插值，所以顶点就是三线性场在该边上的过零点。
- 连通分量：对候选格点做 26 连通标记，每个曲面顶点取其所在网格边的“内侧”端点的分量，每个三角形记三顶点中的最小分量格点数，用于前端隐藏碎片。

| 流场 | 网格 | λ₂ < 阈值 格点占比 | oyf > 0 占比 | 候选格点占比 | 26 连通分量 | 分量格点数中位/90%/99%/最大 | 曲面三角形 |
|---|---|---:|---:|---:|---:|---|---:|
| Channel | 256×192×256 | 3.90% | 45.09% | 1.89%（237,705 点） | 3,404 | 7 / 147 / 983 / 16,393 | 1,075,382 |
| TBL | 276×251×224 | 4.52% | 44.19% | 2.15%（334,328 点） | 3,556 | 7 / 160.5 / 1,515 / 10,684 | 1,481,072 |

一半以上分量不足 10 个格点，但这些碎片只占候选格点约 2%（Channel 232,222 / 237,705 位于 ≥10 格点的分量）。

## 查看器中的呈现

- 侧栏“显示真实标注曲面”下新增：`显示 candidate head 区域（λ₂ < 阈值 ∧ oyf > 0）` 开关（默认关）、候选区域不透明度、`隐藏连通格点数少于 N 的碎片`（默认 1，即不过滤）。图例新增浅绿色 `#7fb069` 半透明面块。
- 选择某个 GT 实例或聚焦某束时，按当前坐标轴范围加载相交的原始分辨率分块（64³ 单元一块，Channel 48 块、TBL 49 块，总计 82 MB，按需加载并缓存），只画三角形重心落在范围内的三角形。
- “全部标注区域”且未聚焦时，加载 vtkQuadricDecimation 简化后的全场网格（目标削减 85%：Channel 161,307、TBL 222,160 个三角形），避免一次向 Plotly 传入百万级三角形。简化网格不带分量信息，碎片过滤对其不生效；提示文字明确标出“简化网格”。
- 每束中心在原始场上插值出 λ₂ 与 oyf，写入 DATA（`center_lambda2`、`center_oyf`、`center_step1`）。点击中心点的详情增加“中心 λ₂ = …（阈值 …），oyf = …；满足 / 不满足 step1 候选条件”。提示区显示当前可见束中 GT 头部束与 Non-hairpin 束的中心满足 step1 的数量。

核对结果：Channel 测试 5,740 束、训练展示 2,220 束，TBL 测试 5,580 束、训练展示 1,740 束，全部中心都满足 step1；包括全部 Non-hairpin 束（Channel 3,925、TBL 4,308）。这符合旧采样链与新增头部采样都要求中心处 λ₂/oyf 的事实：Non-hairpin 束的中心也在候选区域内，二者差别只在人工 GT 归属，不在 step1 条件。

## 构建与证据

- 脚本 `experiments/Build_Task4C_CandidateHeadRegion_1_2.py`（以 `python -m experiments.Build_Task4C_CandidateHeadRegion_1_2` 运行）；前端控件 `experiments/templates/task4c_candidate_head_1_2.js`。脚本先核对原始流场 SHA-256 与 `viewer_package/manifest.json` 一致，再从 1.1 页面备份 `index_1.1.html` 出发生成新 `index.html`，重复运行不会叠加装饰。
- 输出：`viewer/candidate/{index.js, channel_*.js, tbl_*.js, *_overview.js}`、`viewer/candidate_head.js`、`viewer/candidate_head_manifest.json`（含统计、分块哈希、简化率、逐集合中心统计）。`head_coverage_manifest.json` 与 `viewer_manifest.json` 的 html 哈希已更新，工作台 1.5 的 `build_manifest.json` 重新生成（入口不变：`http://127.0.0.1:8767/Other_FMT_AnalysisWorkbench_1.5/index.html#results`）。
- 浏览器检查（2026-09-18，本地 8767 服务）：Channel GT 3 区域加载 4 块、8,913 个三角形，碎片阈值 50 后 8,049 个；全场简化网格 161,307 个三角形；TBL GT 22 区域 8 块、4,546 个三角形；点选详情显示中心 λ₂/oyf；控制台无报错。
- 本版本没有改变任何科学数据。若要把 step1 区域用于新的采样或标签规则，须按 `Task4C_fixed_dataset_rules_v1.md` 建新数据版本，不能沿用本可视化的临时统计。
