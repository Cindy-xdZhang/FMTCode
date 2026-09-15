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
