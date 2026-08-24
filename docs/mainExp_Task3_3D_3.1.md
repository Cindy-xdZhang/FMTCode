# mainExp_Task3_3D_3.1：Task3 论文主表去偏确认

## 预注册目的

Task3 检验固定 Fourier feature encoder（FMT）加入监督神经网络后，是否提高以
Instantaneous Vorticity Deviation（IVD，瞬时涡量偏差）构造的涡/非涡二分类性能。
本版本取代 `mainExp_Task3Universality_2.2` 和 `mainExp_Task3NewFlows_2.3` 作为论文
主表来源；旧版本完整保留为开发证据。

## 为什么必须补跑

旧版 Raw+FMT residual 的 checkpoint 选择先要求 validation F1 比 Raw 高 `.02`，再最大化
Average Precision。虽然没有读取 confirmation label，但该规则显式写入了希望证明的增益。
旧版还缺少同为“冻结 Raw + residual”的 Raw-only 对照。因此 3.1 在查看新 confirmation
结果前冻结以下修改：

1. 每个方法只按自己的 development-validation Average Precision 选择 epoch；
2. F1 threshold 只由该方法自己的 development-validation 数据选择；
3. residual alpha 对所有数据和随机种子固定为 `1.0`，不做网格搜索；
4. 增加 `Raw + Raw-PCA residual`。PCA 仅在 normalized training pathline 上拟合，输出
   268 维；其 residual 网络、训练预算和可训练参数量与 268D FMT residual 完全相同；
5. Raw 主基线只根据 physical-family 的 development-validation AP 在 Raw、Raw-wide、
   Raw-PCA residual 中冻结；confirmation 上不再事后取最大；
6. 使用 5 个新训练随机种子 `30–34` 和每个数据条目 8 个此前未评估过的 confirmation
   起始时间。

## 固定数据和模型协议

- 数据条目：channel observer、half-cylinder Re160/Re640/Re6400、Tangaroa、
  delta-wing resampled/original、F-22、Boeing 747、Smoke buoyancy；共 10 个条目、
  7 个 physical family。
- 标签：`IVD > 0.9 × mean_11x11x11(IVD)`；不得按本实验结果修改。
- primitive：center、x±、y±、z± 共 7 条 pathline。
- pathline：`dt_scale=.25`，积分 48 步，采样 32 点，`16³` seed grid，最大空间维 96。
- FMT：161D 原始 block + 63D time-local Gram-2 + 44D kinematic，共 268D。
- Raw、Raw-wide：端到端训练；Raw-PCA/FMT residual：冻结对应 seed 的 Raw 后训练同构
  residual head。
- 训练：AdamW，learning rate `.001`，weight decay `.0001`，最多 100 epochs，
  patience 20；所有方法使用相同 weighted binary cross-entropy。
- development split：ordinal 0–5 train，6–7 validation；8–9 不参与本版本训练、选择或
  报告。

## 新 confirmation 时间边界

8 个固定起始时间都位于原始时间轴 `[20%,90%)`，且未在旧 development/confirmation
列表中出现。因为部分数据（尤其 Re640）时间轴很短，而一个 primitive 会读取约 14 个
source frames，新窗口不可避免与历史窗口在 source-frame 层面部分重叠。因此本实验验证
的是新起始时间外推，不声称完全不相交的 source-frame 外推。

## 主表规则

- 每个数据条目报告 5 seeds 的 mean ± sample standard deviation；
- 主指标为 F1 score 与 Average Precision（AP，平均精确率）；
- 主增益为 `Raw+FMT residual − development-frozen Raw-only method`；
- 同时完整报告 Raw、Raw-wide、Raw-PCA residual，禁止只隐藏较强 Raw 对照；
- 报每个数据条目、physical-family macro、逐时间片结果和 paired seed bootstrap 95% CI；
- 不设“必须增益 `.02`”的通过条件。负结果原样进入论文表。

## 代码与配置

- `Verify_Task3_FMTClassifier.py`
- `Verify_Task3_FMTResidual.py`
- `Evaluate_Task3_MainTable.py`
- `Build_Task3_Main_Confirmation.py`
- `config/mainExp_Task3_3D_3.1_*.yaml`
- `ibex_bash/mainexp_task3_3d_3.1_*.sh`

## 结果

等待 Ibex A100 实验完成后填写；不得用旧 confirmation 数字预填。
