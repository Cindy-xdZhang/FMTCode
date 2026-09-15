# Task4-c：涡线簇分类的三维查看器

版本`Other_Task4C_BundleVisualization_1.1`，2026-09-15用户要求：在原三维流场内显示vortex line bundle的Hairpin / Non-hairpin预测，同时叠加半透明ground truth曲面。此版本只读取已有预测，不训练模型，不修改Task4-c定义、数据或指标。

## 输入和真实坐标

使用4.14保存的`physical/{channel,tbl}/{train,test}/geometry.npy`、`metadata.npz`和相应已完成运行的`predictions.npz`。通过`x = geometry × radius + centroid`逐束恢复流场坐标，排除补零的无效线，每条有效线完整保留32点。检查恢复与逆运算、预测与标签/头区/尺度行序、模型实际输入的元数据哈希，以及完整集合F1复算。浏览器坐标转float32的误差单独检查；此转换不参与分类。

全部展示模型使用固定seed96611，不根据分数选择种子。首批包括原FMT 4.14，以及32³/36³/48³小Conv3D 4.22中已完成的模型；尚未产生`result.json`的模型在页面列为未载入，向新目录重新执行export，再执行build即可添加。不会使用未完成checkpoint推理，也不复用用户已纠正的24³旧逐样本预测来推导0.7499。

每流场、每集合按中心位置在各坐标轴范围归一化后的最远点次序抽取240束，顺序固定且不读取标签或预测。默认显示前100束，可调显示数量、单独显示两类及聚焦局部区域。这里的坐标轴归一化只用于选择展示样本，渲染坐标及长宽高比例保持原物理尺度。多个模型使用相同展示样本；F1始终显示该模型对该流场完整集合的指标，不是显示子集的F1。

## Ground truth与显示

本地原文件`channel_flow/channel_GTs.vtk`和`tbl_flow/tbl_GTs.vtk`需与冻结配置哈希一致。以`vtkDataSetSurfaceFilter`提取原标注体单元的真实边界，并三角化显示。保留所有有效`VortexIds`，包括Channel编号0；不使用新的lambda2等值面替代标注，不平滑、不减少三角形、不根据预测裁切GT。

线簇预测Hairpin为青绿`#218990`，Non-hairpin为柔和棕色`#c08650`；GT统一灰紫`#a6a0b4`，默认不透明度18%，可调0–55%。同一线簇内所有有效线使用同一类别颜色。页面支持Channel/TBL、训练/测试、模型、标注实例切换，旋转缩放、点击中心查看概率和真实束标签、聚焦线簇。地面位于原流场zmin；x=streamwise，y=spanwise，z=vertical。

标签来自候选头区的实例归属，整个线簇可能伸出GT曲面；不能据此将单个线段在曲面外直接判为分类错误。图展示已有采样束的分类，不表示稠密全流场预测，也不增加独立实例泛化证据。

## 运行与交付

代码`experiments/Visualize_Task4C_Bundles_3D.py`，模板`experiments/templates/task4c_bundles.html`，配置`config/Other_Task4C_BundleVisualization_1.1.json`。导出只需NumPy，构建需要项目已有VTK和Plotly。

```powershell
# Ibex：在冻结数据和已完成预测仍可读取的目录运行
python -m experiments.Visualize_Task4C_Bundles_3D export --output outputs/Other_Task4C_BundleVisualization_1.1/package

# 将这个小型展示包带回本地后，构建完全离线的HTML与VTK几何文件
.\tmp\jhtdb-venv\Scripts\python.exe -m experiments.Visualize_Task4C_Bundles_3D build --package outputs/Other_Task4C_BundleVisualization_1.1/package --input-root "C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D" --output outputs/Other_Task4C_BundleVisualization_1.1/viewer --preview
```

交付一个可离线打开的`index.html`、两个GT曲面VTP，以及Channel/TBL训练/测试各一个线簇VTP。每个线簇VTP同时存放全部可用模型的`p_<model>`和`class_<model>`单元属性、原样本行号、真实束标签与尺度，可在VTK软件中切换着色。不下载或保存模型权重。HTML预览只供交互查看与检查，不当作论文最终排版图。

导出清单保留配置、源文件与预测哈希、抽样规则和行号；构建清单保留GT与HTML哈希、模型版本、预览相机。真实数据的显示与复算检查集中于正式入口，不新增test/verify脚本。

## 实际交付与检查

CPU导出作业51906466，科学commit`d5b778f98b0c920257e47663b5e8448318e71eaf`，2026-09-15T11:19:19 UTC完成，退出0。首版包含FMT4.14和32³小Conv4.22的seed96611，36³、48³尚未在此次导出时完成。四个分区各240束，共960束、16,981条有效线，全部VTP顶点、有效线数量、样本行号、两个模型的逐线概率和二分类颜色独立复核PASS。GT边界Channel313,567三角形/74实例（含0），TBL317,232三角形/58实例，无简化。

输出`outputs/Other_Task4C_BundleVisualization_1.1/viewer/index.html`，内嵌Plotly和全部数据约36.8 MB，另有六个VTP和两个VTK预览PNG。源证据`package/manifest.json`，检查`code_checks.json`与`geometry_audit.json`，渲染清单`viewer/viewer_manifest.json`。浏览器访问`http://127.0.0.1:8896`被自动审批拒绝，理由是尚未明确授权该本机origin访问；交互检查等待授权，不能将源码和VTK检查写成浏览器验证通过。页面可由用户直接打开本地HTML。

## 1.2：完整集合与全部Hairpin（2026-09-15）

用户指出Hairpin数量最多28。核对1.1展示包，Channel训练240束中两个已载入模型各预测28束Hairpin；完整13,500训练束中，FMT预测2,681束、小Conv32³预测2,682束。这是查看器导出抽样造成的上限，不是训练数据只有28束。此前仅增加独立滑块，没有扩展导出范围，因此未满足大量查看Hairpin的需求。

新增配置`config/Other_Task4C_BundleVisualization_1.2.json`，`selection=all_source_rows`，直接导出原训练/测试集合的每一行，合计37,000束（两个流场各13,500训练＋5,000测试），不重新积分、不复制曲线、不调整阈值或预测。主脚本和模板继续共用，不新增test/verify脚本。旧1.1数据包保留；新包位于`outputs/Other_Task4C_BundleVisualization_1.2/package`。

页面默认选择全部预测Hairpin、100束Non-hairpin，新增“全部Hairpin”“全部Non-hairpin”及直接输入数量。上限是当前区域中实际预测类别的总数；全区域时覆盖完整集合。所有模型使用完全相同几何行，F1仍来自完整集合。预测Hairpin与真实Hairpin标签明确区分，不能为了多显示将负预测改成正预测。

完整坐标每128束分成一个本地JavaScript文件，保留每条有效线的全部32点，去掉补零槽位。HTML先提供全部标签、概率和空间中心，显示所需几何才按需载入；同一分片的重复请求复用。用户需保留页面同目录的`geometry`子目录。全部文件位于本地，不使用网络CDN，不需要本机HTTP服务。新版本不默认输出37,000束的大型VTP；旧版六个VTP保留，新的两个GT边界VTP仍输出。

纯JavaScript检查包含语法、真实几何分片长度、并发请求复用、最后不足128束分片、全部Hairpin及两类独立数量选择。实际浏览器交互检查仍未执行，不能以脚本检查代替浏览器验证。

科学提交`5307bf3a849d07625dac21d92e3abb4a35778b44`，配置SHA256 `23c694e50428d55721ee83f649a0155b7e74922e7e4a0330bc4ef0f462d6784b`；CPU导出作业51907655，2026-09-15T12:34:30 UTC提交。实际完成与几何全量复算随后写入实验日志和Ibex登记。

### 1.2 完成记录

CPU作业51907655在cn604-10完成，实际UTC运行12:34:34.113378至12:34:44.418508，Slurm为COMPLETED/0:0。全量37,000束包含680,519条有效线，分为292个几何文件；HTML约30.14 MB。全部原样本行覆盖一次，逐束float32坐标、每条线全部32点、标签与三个模型概率逐元素核验通过；12个流场/集合/模型组合的混淆矩阵和F1与冻结结果一致。

| 模型（seed96611） | Channel训练 Hairpin | Channel测试 Hairpin | TBL训练 Hairpin | TBL测试 Hairpin |
|---|---:|---:|---:|---:|
| FMT 4.14 | 2681 | 863 | 1939 | 466 |
| 小Conv3D 32³ | 2682 | 935 | 1937 | 573 |
| 小Conv3D 36³ | 2682 | 928 | 1937 | 522 |

48³在此次导出时尚未完成，未展示或补造预测。Node执行正式加载器验证全部2,681束Channel训练预测Hairpin可被选择并加载，最后不足128束分片及共享请求均通过；这仍是代码检查，未执行浏览器交互验证。

交付`outputs/Other_Task4C_BundleVisualization_1.2/viewer/index.html`及同目录`geometry/`。原1.1页面已设为指向新版的相对路径跳转，原完整HTML和清单位于原viewer的`history/`目录，旧包未改。核验依据：新输出目录中的`full_geometry_audit.json`、`full_loader_checks.json`、`runtime_events.jsonl`和`viewer_redirect.json`。新版HTML SHA256为`958bce7639143592eba825c2eec1e9796576bcd3def2ae58988005bd48c1af27`。
