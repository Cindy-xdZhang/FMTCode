# mainExp_Task5_3D_1.1：3D 不同尺度几何学习

## 研究问题

Task5 将 Task3 的 fixed-scale pathline primitive 改为 variable-scale primitive。
IVD 标签、网络结构、FMT 268D 表示、Raw-PCA 268D 强对照、训练预算和模型选择规则
均沿用 `mainExp_Task3_3D_3.2_global_ivd`。唯一方法变化是每个 primitive 的：

1. 中心到 `x±/y±/z±` 邻居的初始距离；
2. RK4 数值积分步长；
3. 积分步数。

3D primitive 仍为中心加六邻居共7条线。每条线积分后统一重采样为32点，因此网络输入
恒为 `7×32×3`。用户描述中的5条线对应2D cross；在3D主实验中改成5条线会同时删除
一个空间轴，无法把结果只归因于尺度变化。

## 冻结的尺度拆分

- train：18个tuple，空间 offset 为`0.25/0.50/1.00`倍最小网格间距；6种
  `(dt_scale, integration_steps)`组合，总时间跨度为4–12个source frame。
- validation：6个未在train出现的tuple，offset为`0.375/0.750`。
- confirmation：9个train/validation均未出现的tuple，offset为
  `0.33/0.67/1.25`；其中`1.25`是空间尺度外推。
- 每个时间片内用固定随机种子打乱空间seed，再均衡分配尺度，任意tuple的分配数量差不超过1。

development使用4个训练时间片和2个validation时间片；confirmation使用4个更晚的
时间片。所有pathline source window的最大前视范围为12个source frame，development与
confirmation窗口不重叠。标签始终为`IVD(seed) >= percentile95(IVD volume)`。

## 论文主比较

1. fixed-scale Task3 Raw-PCA residual直接迁移到variable-scale confirmation；
2. fixed-scale Task3 Raw+FMT residual直接迁移；
3. Task5 variable-scale Raw与Raw-wide；
4. Task5 variable-scale Raw-PCA residual；
5. Task5 variable-scale Raw+FMT residual。

checkpoint仅按各自development-validation Average Precision选择，threshold仅用
development冻结，residual alpha固定为1.0。5个训练seed为`40–44`，便于与fixed-scale
Task3做paired比较。最终输出总体、逐流场、physical-family和9个confirmation尺度tuple
的F1与Average Precision。

## 代码与配置

- `FMT_Utils/MultiscalePathline_3D.py`
- `Build_Task5_Multiscale_Cache.py`
- `Evaluate_Task5_Multiscale.py`
- `config/mainExp_Task5_3D_1.1.yaml`
- `config/mainExp_Task5_3D_1.1_*.yaml`

结果在Ibex任务完成后追加，未完成前不得写方法结论。
