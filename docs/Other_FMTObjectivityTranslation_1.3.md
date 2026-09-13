# Cylinder 后半程客观性图（1.3）

用户2026-09-06更新：所有cylinder数据忽略原始模拟前50%，起始t>=7。
规则保存在`config/cylinder_time_policy.json`，项目默认说明在`AGENTS.md`。
当前原始模拟时段均为[0,15]，因此采用t>=7.5；Re640文件本身已裁到[7.5,15]，不再次截半。

## 时间选择

| 数据 | 原始模拟范围 | 文件范围 | 首个合规确认缓存 | 本次初始时间 |
|---|---|---|---|---|
| Re160 | 0–15 | 0–15 | ordinal2/index105 | 10.5 |
| Re640 | 0–15 | 7.5–15 | ordinal0/index20 | 9.5 |
| Re6400 | 0–15 | 0–15 | ordinal2/index105 | 10.5 |

仅依据manifest的时间元数据确定，不依据涡标签、图像或指标择优。
已有缓存没有更早的合规时间片，选中时间符合下限而不是声称恰从t7.5开始。
后续重新生成数据可从t7.5或其后第一个源帧开始，必须记录新缓存/实验版本。
原冻结训练、校准、确认划分及旧图片不追溯改写。

## 方法与导出

继承1.2的数学定义与验证：完整源网格空间体积均值作为时变纯平移observer速度；自然样条插值速度并积分位移；0/25/50/100%四档分别积分observed field并重新分类。
Task1-4.1固定配方、seed7080、一次拟合；整条中心轨线按涡/非涡着色，六条邻居线参与编码。
同一流场四档同物质种子、相机和物理范围；全部共同有效轨线保留。
非涡线alpha0.12、涡线alpha0.95，四档一致；论文和汇报版分别最小字号7/13pt。
完整参数见`config/Other_FMTObjectivityTranslation_1.3.json`。

```bash
python experiments/Visualize_FMT_Objectivity_Translation_1_3.py \
  --dataset cylinder3d --global-mean \
  --time-policy config/cylinder_time_policy.json \
  --output-dir outputs/Other_FMTObjectivityTranslation_1.3/cylinder3d
```

数据名还支持`halfcylinderRe640`和`halfcylinderRe6400`。选择逻辑在`FMT_Utils/CylinderTimePolicy.py`。
远端如缺PyMuPDF，可显式`--defer-pdf-collision-audit`，但必须下载最终PDF后完成本地审计才能交付。
每组导出paper/slides的PNG/PDF/SVG及论文TIFF。仅下载图片，预测数组留远端。
两项新增时间规则测试通过；数值结论仅记录在`docs/experiment_log.md`，作业状态在`docs/ibex_run_registry.md`。

Re6400源文件仅在本机：Prepare_Cylinder6400_Observer_Window_1_3.py提取原索引105–118和冻结空间步幅7/3/1，原始时间坐标与索引保留，其余帧明确为缺失。observer均值在原始640×240×80完整网格上计算后另存，绝不用降采样网格替代均值。运行传入--source-file及--observer-json；代码验证窗口文件SHA。原始逐帧SHA、提取规则及完整均值保存在source_window/observer.json，输出图元数据也携带其身份。

已完成三组×两套版式，共21个图片文件。最终6份PDF字体最小7/13pt、碰撞0FAIL，6份面板尺寸检查PASS；每图4项白色背景边缘WARN已逐栏目视接受。图包Cylinder_late_time_observer_figures.zip位于输出根目录；图片哈希清单image_manifest.json，分类结果和证据边界见experiment_log.md。
