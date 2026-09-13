# Cylinder：七档平均平移 observer（1.5）

本版按用户要求将一列四图扩展为一列七图，速度系数依次为0、1/6、2/6、3/6、4/6、5/6、1；图标题显示0%、16.7%、33.3%、50.0%、66.7%、83.3%、100.0%。
七个系数作用于同一个完整原始空间网格的平均速度随时间曲线 b(t)。时间方向用自然三次样条插值速度，再积分同一个样条得到位移。

`u_alpha(t)=alpha*b(t)`；`d_alpha(t)=integral(u_alpha,t0,t)`；观察坐标为`y=x-d_alpha(t)`；observed field在原坐标`y+d_alpha(t)`处读取速度，再减去`u_alpha(t)`。
所有七档均独立积分observed field并重新计算FMT特征和分类，不在标签之间插值。

| 数据 | 初始时间 | 中心轨线数量 | 显示时长 | 分类窗口 |
|---|---:|---:|---:|---|
| Re160 | 10.5 | 7220 | 96步，约2.4 | 前48步的原32点 |
| Re640 | 9.5 | 7154 | 同上 | 同上 |
| Re6400 | 10.5 | 7242 | 同上 | 同上 |

精确复用1.4保存的物质种子、速度曲线、原始源文件和时间窗口；不新增种子、不根据结果筛选。每档七线primitive全部完整才允许输出，并要求数量与1.4完全相同。
固定Task1 `mainExp_Task1_3D_4.1` 的 `fmt_all+kin4`、PCA8、seed7080分类器，训练/校准划分不改。0/50/100%三档必须与1.4逐条标签一致、坐标最大差低于1e-10，否则终止。
每条显示轨线沿用其前48步primitive的二分类标签，不表示延长尾段每个位置都独立分类。不附用原缓存队列的IVD F1；分类变化的科学结论只记录在experiment_log.md。

论文版7.2×17.085英寸，PPT版10.8×25.6275英寸，最小字体7/13pt。保持上一版每栏物理高度、统一相机与七档共同物理边界；红色为涡、蓝色为非涡。继承1.4的完整轨线投影修复`set_clip_on(False)`。
论文/PPT均导出PNG、PDF、SVG，论文另有600dpi TIFF。七联图是纵向长图；插入常规横向PPT时可逐栏裁切展示。

| 绘图方法 | 代码 | 特点 |
|---|---|---|
| 七档observer纵向对比 | `experiments/Visualize_FMT_Objectivity_Translation_1_5.py` | 同物质轨线、冻结分类器、速度等间距七档；独立计算；两套尺寸；共用范围与相机 |
| 七档独立积分与分类 | `FMT_Utils/SevenTranslationObservers_3D.py` | 独立版本，保留旧四档实现；检查轨线变换一致性和跨档完整性 |

配置：`config/Other_FMTObjectivityTranslation_1.5.json`；提交：`ibex_bash/other_fmt_objectivity_seven_1p5.sh`。科学数组仅保存在Ibex；本地只下载图片，保存本地图片审计报告。1.4代码和结果保留。

最终输出已完成：outputs/Other_FMTObjectivityTranslation_1.5/<dataset>/final，三个流场共21份图片；总包Cylinder_seven_observer_figures.zip。六份七栏尺寸审计PASS，最终PDF字号7/13pt、碰撞0FAIL。每份7个统计文字跨白色axes背景边缘WARN已逐栏复核接受，全部42栏和六张整图检查完成；保留自动报告与目视说明。具体实验计数及结论见experiment_log.md的1.5条目。
