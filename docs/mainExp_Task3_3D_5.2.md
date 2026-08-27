# mainExp_Task3_3D_5.2：第三空间 population 最终确认

## 预注册状态

尚未运行，尚未生成最终 cache，尚无性能结果。本文件只固定最终边界；运行结果无论
成功或失败都将在此追加，不能覆盖 `mainExp_Task3_3D_5.1` 的失败记录。

## 冻结条件

- 主比较：FMT residual 与同宽、同结构、同训练样本、同优化器、同 seed 的
  train-only Raw-PCA residual。
- 标签：standard whole-field IVD p95。
- 数据条目：channel、half-cylinder Re160/Re640/Re6400、Tangaroa、delta-wing
  resampled/original LBM、F-22、Boeing 747、Smoke buoyancy，共 10 条目、7 个
  physical family。
- 最终物理时间与 4.1/5.1 相同；空间相位固定为
  `[0.318359375,0.4561042524005485,-0.3352]`。
- 最终 4 切片/条目，paired training seeds 40--44。
- 目标：dataset-macro absolute F1 gain `>= +.15`。Average Precision、逐条目、
  family-macro 与逐 seed 结果同时报告。
- Stage 2 selection 未冻结时，builder 必须拒绝生成 cache；selection 还必须明确
  声明 4.1 为 exposed training、5.1 为 exposed validation。

完整开发协议见 `docs/Verify_Task3_SpatialRobust_5.2.md`。
