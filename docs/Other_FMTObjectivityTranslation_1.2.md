# Cylinder Re160：全局平均平移 observer

## 定义与证据范围

本图回答：从原始参考系逐步移动到随全局平均流运动的参考系时，固定Task1分类器给同一组物质轨线的标签是否改变？不预设标签不变。

目标速度 `b(t) = integral_domain(v(x,t) dV) / integral_domain(dV)`。
在完整源网格160×60×20、每个原始物理时刻计算；采用张量积梯形积分权重。
三分量任一缺失时联合排除该节点，保留所有有效零速度节点；无可靠固体掩膜时不猜测固体范围。
时间范围为既定确认ordinal0的源窗口，起始索引39；实际轨线时长沿用48步、步长0.25倍数据帧间隔。
这是空间上的全局均值，随时间变化，不是从轨线种子位置或涡标签估计的速度。

四档 `u_alpha(t)=alpha*b(t)`，alpha=0、0.25、0.5、1。
对速度作自然三次样条插值，同一条样条积分为 `d_alpha(t)`；
`y=x-d_alpha(t)`，`observed(y,t)=v(y+d_alpha(t),t)-u_alpha(t)`。
整体平移分量被扣除，空间变化的背景剪切等仍保留。

原始场积分沿用冻结缓存的降采样网格80×60×20；仅observer均值使用完整源网格。
四档同物质种子、同采样时间、同分类器，七条primitive轨线都变换并重新编码，整条中心轨线着色。
固定Task1-4.1配方fmt_all+kin4、主成分分析8维、seed7080；训练ordinal0–7、校准8–9。
仅原始训练/校准集合拟合一次，不重新选特征、模型或簇映射。
确认ordinal0仅显示和评估，不据它优化observer。

## 代码与运行

- `experiments/Visualize_FMT_Objectivity_Translation_1_2.py`：四联图、冻结分类器、结果和审计。
- `FMT_Utils/GlobalMeanObserver_3D.py`：完整源网格空间均值。
- `FMT_Utils/TranslationObserver_3D.py`：坐标变换、观测场积分及一致性检查。
- `config/Other_FMTObjectivityTranslation_1.2.json`：本次冻结绘图约定。
- `ibex_bash/other_fmt_objectivity_mean_1p2.sh`：CPU作业。

```bash
python experiments/Visualize_FMT_Objectivity_Translation_1_2.py \
  --dataset cylinder3d --global-mean --device cpu \
  --output-dir outputs/Other_FMTObjectivityTranslation_1.2/cylinder3d
```

输出paper/slides的PNG/PDF/SVG及论文TIFF；不保存模型。
完整轨线及逐图JSON保留远端，仅下载图片。JSON包含均值、位移、分类改变数、共同有效样本、固定瞬时涡量偏差（IVD）p95参考下的F1分数。
不变性和识别准确率是不同问题，不能把标签一致直接解释为预测正确。

## 验证与图版

9项数值测试通过：平均速度权重、缺失值联合排除、速度积分、逆位置读取、observed积分与坐标变换等价、域边界、独立分类及时间外推检查。
一列四图由用户明确指定；四栏同相机和物理尺度，范围取所有四档轨线并集。
每栏显示所有共同有效中心轨线，因积分越域而剔除的primitive在JSON逐项记录。
颜色为红=涡、蓝=非涡；字体论文至少7pt、汇报至少13pt。
单个冻结种子/时间片的确定性分类计数，不展示跨实验置信区间；源代码“mean/seeds但无误差条”警告不适用于该图。
最终数值结论只写入`experiment_log.md`，作业状态写入`ibex_run_registry.md`。

最终文件位于outputs/Other_FMTObjectivityTranslation_1.2/cylinder3d/final；图片压缩包位于其上层。计算作业51401240已保存完整结果，因远端缺少PDF审计脚本中止；渲染恢复作业51401324完成，最终PDF碰撞检查已在本机完成。两套面板检查PASS、字号7/13pt、碰撞0FAIL。各4个白色背景边缘WARN已逐栏目视接受。最终全部轨线保留，非涡/涡透明度0.12/0.95在四档固定，相机zoom2.1相同。结果与证据边界见experiment_log.md。
