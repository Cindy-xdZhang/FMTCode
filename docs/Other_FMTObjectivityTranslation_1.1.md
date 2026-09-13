# 客观性验证图：平移Killing observer

新代码：[绘图与Task1接口](../experiments/Visualize_FMT_Objectivity_Translation_1_1.py)、
[参考系变换与轨线积分](../FMT_Utils/TranslationObserver_3D.py)、
[绘图约定](../config/Other_FMTObjectivityTranslation_1.1.json)。
独立于原来的三联图模式a/b，本图固定为**一列四张子图**。

已阅读用户指定理论文档的实际文件
`C:/Users/xingdi/sources/optimal-connection/docs/referenceframe/referenceFrame_overview_zh.md`的第二节；
原提供的嵌套路径不存在。本实现明确区分相对速度与经过坐标变换的observed field。

## 数学约定

纯平移Killing observer指空间上处处相同、可随时间变化的参考系速度，空间梯度为零。
目标速度为原坐标中的`b(t)`，四档`alpha = 0, 0.25, 0.5, 1`：

```text
u_alpha(t) = alpha * b(t)
d_alpha(t) = integral from t0 to t of u_alpha(s) ds
x_observed = x_original - d_alpha(t)
Q(t) = I, c(t) = -d_alpha(t)
v_observed(y,t) = v_original(y + d_alpha(t),t) - u_alpha(t)
```

因此先将查询点`y`拉回原始位置`y+d_alpha(t)`，在那里读取原场，再减observer速度。
不能直接读取`v_original(y,t)-u_alpha(t)`，也不能将瞬时速度直接当位移。
时变输入先对**速度**作自然三次样条插值，再积分同一条样条；不额外拟合或插值位移。
四档只是同一目标速度的比例，不是对最终粒子位置、图像或分类标签插值。

四档均直接在各自observed field中作四阶Runge–Kutta（RK4）积分。
同时比较`observed_path(t)`和`original_path(t)-d_alpha(t)`；超出指定容差时停止出图，
提示减小积分步长。原始场的域外查询返回NaN；observer中负坐标本身不代表越域，
必须在拉回后的原坐标判断。所有observer初始变换为恒等变换。

## 分类与显示

一个primitive包含中心轨线及六条邻居轨线。四档变换和积分全部七条线，重新计算FMT，
将该primitive的涡/非涡二分类结果着色到其**整条中心轨线**。
红线为涡，蓝线为非涡，不是只在种子位置画分类散点。
默认显示所有共同有效primitive，四栏使用同一物质种子、相机、坐标范围与空间比例；
范围取全部四档中心轨线的并集，不逐栏重新居中抵消平移视觉效果。

Task1命令行接口只在原始训练集拟合一次标准化、主成分分析和KMeans；仅用原始校准集确定匿名簇对应关系。
随后四档共享该拟合结果，各自重新编码和预测，不重新聚类、不重新校准、不复制0%标签。
现行配方中的中心轨线时间差分本身不能视为任意时变平移下不变性的证明，
本图会如实报告标签变化数、变化比例和特征最大绝对变化，不预设验证通过。

出现积分不完整的primitive时，在分类之前对四档取共同有效集合，保存原索引和排除数。
不按预测、IVD得分或图像效果筛选轨线。该结果只检验本次纯平移，不证明旋转或一般observer客观性。

## 运行

以下是**参数示例**，`1 0 0`不是用户指定的正式目标observer。运行环境需要项目原有PyTorch、
scikit-learn、PyYAML、NumPy、SciPy、netCDF4、Matplotlib及PDF审计依赖PyMuPDF。
默认Task1配方4.1、seed7080、首个确认片ordinal0，保持当前基线不变。

```powershell
python experiments/Visualize_FMT_Objectivity_Translation_1_1.py --dataset cylinder3d --target-velocity 1 0 0 --output-dir outputs/Other_FMTObjectivityTranslation_1.1/cylinder3d_velocity_example
```

`--selection`可指定本地归档的`selected_config.json`（例如带`_ibex`后缀目录）；
`--source-file`可指定原始NetCDF。选择文件必须与协议配置哈希一致。
`--substeps 2`表示模型原始积分步长再细分2次；
`--correspondence-tolerance 1e-5`为轨线一致性的绝对物理距离容差，可按实际单位预先设置。
不覆盖已有非空输出目录，不保存训练模型。

时变observer使用`--observer-json observer.json`。格式例：

```json
{
  "time_origin": "relative_to_seed",
  "times": [0.0, 0.6, 1.2, 2.0],
  "velocities": [[1, 0, 0], [1.2, 0.1, 0], [0.8, 0.2, 0], [1, 0, 0]]
}
```

`times`与速度必须使用原场物理单位，并完整覆盖积分区间。`time_origin: physical`则表示绝对物理时间。
超出时间表范围报错，不静默外推。需要Task2/3等其他固定分类器时，调用
`evaluate_observers(lab_velocity, initial_primitives, times, classifier, observer, ...)`再调用`render()`；
`classifier`必须接收`[N,7,L,3]` observed轨线并返回`[N]`二分类，可额外返回特征矩阵。

## 输出与验证边界

输出`paper/slides`各自PNG、PDF、SVG；论文另有600dpi TIFF。PDF/SVG中的栅格轨线也使用明确的400/220dpi。
`observed_pathlines.npz`保存四档七线轨迹、中心线分类、原primitive索引和实际时间。
逐图JSON记录速度样本、积分约定、分类变化、轨线一致性、域外排除以及数据和代码身份。
面板尺寸、PDF字号、文字碰撞自动检查；WARN仍需查看实际输出。

本次完成7项数值测试及两套版式的解析场渲染检查。渲染检查使用明确标为fixture的简单阈值回调，
仅用于核查线条着色、标签独立计算与排版，**不是FMT实验结果**。
两份PDF最小字号7pt/13pt、碰撞0 FAIL；各4个WARN均为面板统计文字跨白色绘图区背景边缘。
已逐栏查看，文字完整、不遮挡轨线，与背景均为白色的有意布局一致，接受这些WARN并保留原审计状态。
Task1真实数据接口已接入并核对源码；当前本机测试环境无PyTorch，未运行真实FMT分类端到端验证。
尚未指定正式目标observer，未提交Ibex作业，也未产生新的客观性实验结论。
