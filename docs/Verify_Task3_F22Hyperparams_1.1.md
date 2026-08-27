# Verify_Task3_F22Hyperparams_1.1

## 目的

修复 `mainExp_Task3_3D_3.2_global_ivd` 中 F-22 的反例。旧配置使用
268 维混合 FMT feature，在 8 个 confirmation 时间片上相对同结构
Raw-PCA residual 的 F1/AP 分别低 `0.02689/0.01292`。

本实验只回答：F-22 是否需要与其输运尺度匹配的 FMT feature block 和
residual 训练超参数；标签仍是 whole-field IVD p95，网络主体和 temporal
split 不变。

## 搜索协议

- development cache 共 10 slices：ordinals 0–5 训练，6–7 validation，
  8–9 只在候选冻结后作 outer-development。
- Stage 1 搜索 14 个 FMT blocks，保留 `kin2/kin6/kin4`。
- Stage 2 搜索 3 个 blocks × 10 个 residual/width/learning-rate/fusion
  组合，每个候选使用 seeds 40–42。
- 每个 FMT candidate 都与相同 auxiliary width、相同 residual 结构和训练
  设置的 train-only Raw-PCA residual 配对。
- 候选按 validation 上的 FMT−Raw-PCA F1 排名，以 Average Precision 和
  最差 seed F1 gain 打破并列。
- 8 个 fresh-confirmation 索引
  `[34,47,60,74,89,104,117,141]` 在 Stage 2 selector 完成前冻结；其
  schedule SHA-256 为
  `de6ad5a535a4c2bbadd180d9bc6db2a25ebd078721e67cb8d0e0f05f35b61ada`。

## 冻结配置

- Feature：`kin6`，即对 pathline primitive 的局部速度梯度序列计算前
  6 个离散 Fourier 频率的无参数 kinematic features，输出 44 维。
- Residual input：`geometry_fmt`。
- Auxiliary hidden width：64。
- Learning rate：`3e-4`。
- Fusion/threshold/checkpoint：全部只由 development validation 冻结。
- Selection SHA-256：
  `46aa4f46fdc48427105de9a90843e128388be53769129b6947143e73d166cb4f`。

## 结果

| 数据层级 | FMT F1 | Raw-PCA F1 | F1 gain | FMT AP | Raw-PCA AP | AP gain |
|---|---:|---:|---:|---:|---:|---:|
| Development validation, 3 seeds | .88785 | .86708 | +.02077 | .94302 | .91428 | +.02874 |
| Outer-development ordinals 8–9 | .84088 | .81773 | +.02314 | .91090 | .86960 | +.04131 |
| Fresh 8 slices | .89371 | .89368 | +.00003 | .94425 | .94507 | −.00082 |

Fresh confirmation 对论文常规 no-FMT baseline 的结果：

| Seed | FMT F1 | Strong Raw F1 | Gain | FMT AP | Strong Raw AP | Gain |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | .88759 | .87172 | +.01587 | .94169 | .93580 | +.00589 |
| 41 | .89047 | .87573 | +.01474 | .94295 | .92176 | +.02119 |
| 42 | .90308 | .88138 | +.02170 | .94812 | .93475 | +.01337 |
| Mean | .89371 | .87628 | **+.01744** | .94425 | .93077 | **+.01349** |

其中 Strong Raw 是同一 seed 的 Raw 与 Raw-wide 两个 no-FMT 网络中均值更强
的一臂；本实验中两项均为 Raw-wide。FMT 对 strong Raw 的 3/3 seeds 的
F1 与 AP 均为正。

## 结论边界

1. 本实验支持 Task3 的标准命题：在 F-22 whole-field IVD p95 二分类上，
   加入 FMT 的网络优于不使用 FMT 的 Raw/Raw-wide 网络。fresh 8-slice
   mean F1/AP 增益为 `+.01744/+.01349`。
2. 本实验不支持更强命题“FMT 明显优于所有 Raw-only 特征扩展”。对同结构、
   同宽度 Raw-PCA residual，fresh F1/AP 差为 `+.00003/−.00082`，应解释为
   持平；不能使用 positive development/outer 数字覆盖这个结果。
3. 与 `mainExp_Task3_3D_3.2_global_ivd` 相比，F-22 从相对 strong Raw 的
   不明显提升变为 3/3 seeds 稳定正增益，说明 F-22 需要 family-specific
   kinematic feature block；这不授权修改其他 flow 的冻结配方。

## 可追溯入口

- 搜索：`Prepare_Task3_F22_FocusedSearch.py`、
  `Search_Task3_FMTResidual_Stage2_3D.py`
- Fresh cache：`config/Confirm_Task3_F22Hyperparams_1.1_cache.yaml`
- 冻结评估：`Evaluate_Task3_F22_FocusedConfirmation.py`、
  `config/Confirm_Task3_F22Hyperparams_1.1_evaluate.yaml`
- Ibex jobs：50922717、50922718、50922719、50923816
- 本地结果：
  `outputs/Verify_Task3_F22Hyperparams_1.1/` 与
  `outputs/Confirm_Task3_F22Hyperparams_1.1/final_confirmation/`
