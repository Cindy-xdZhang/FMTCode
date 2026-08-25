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

## Ibex 执行方式

经用户允许，论文主表统一改在 32 GB V100 上运行并记录实际型号。baseline 保持两个
独立 group child；residual 按 `10 datasets × {FMT, Raw-PCA}` 拆成 20 个互不共享输出
文件的 array shard。每个 shard 独立写 CSV/checkpoint，全部完成后
`Merge_Task3_ResidualShards.py` 校验每个 `(dataset, method)` 恰有 5 个 seed、无重复且
checkpoint/history 数量完整，再原子生成 evaluator 读取的合并目录。该并行化只改变
调度和文件组织，不改变数据、训练、选择或评估协议。

## 结果

Ibex V100 已于 2026-08-24 完成。10 个数据条目均包含 5 个训练 seed 和 8 个
confirmation 起始时间。审计文件确认 confirmation 数据未用于方法、checkpoint、alpha 或
阈值选择；7 个 physical family 在 development-validation 上都选择了
`Raw-PCA residual` 作为强 Raw 对照。

| 数据条目 | Raw-PCA F1 | Raw+FMT F1 | F1 增益 | Raw-PCA AP | Raw+FMT AP | AP 增益 |
|---|---:|---:|---:|---:|---:|---:|
| Boeing 747 | .6808±.0014 | .7301±.0043 | +.0494 | .7050±.0033 | .8101±.0040 | +.1051 |
| Channel observer | .8294±.0041 | .8652±.0038 | +.0358 | .9224±.0028 | .9443±.0013 | +.0219 |
| Half-cylinder Re160 | .7669±.0062 | .7313±.0071 | -.0356 | .8205±.0169 | .7853±.0178 | -.0352 |
| Delta-wing original LBM | .7056±.0026 | .7416±.0009 | +.0360 | .6767±.0025 | .7701±.0015 | +.0934 |
| Delta-wing resampled | .7049±.0011 | .7409±.0014 | +.0359 | .6710±.0025 | .7636±.0015 | +.0926 |
| F-22 | .7146±.0091 | .6634±.0151 | -.0511 | .7809±.0145 | .7020±.0201 | -.0789 |
| Half-cylinder Re640 | .8200±.0063 | .7771±.0060 | -.0430 | .8944±.0060 | .8537±.0087 | -.0408 |
| Half-cylinder Re6400 | .7741±.0102 | .7603±.0041 | -.0137 | .8589±.0093 | .8355±.0065 | -.0234 |
| Smoke buoyancy | .7113±.0050 | .6900±.0060 | -.0212 | .7712±.0032 | .7557±.0040 | -.0156 |
| Tangaroa | .9346±.0020 | .9332±.0015 | -.0014 | .9823±.0012 | .9799±.0015 | -.0025 |

### 结论

需要把两个不同强度的比较明确分开：

1. **相对原始 Raw baseline，Task3 命题得到支持。** Raw+FMT 在 10/10 条目的 F1 和
   AP 上都更高；条目平均增益为 F1 `+.0701`、AP `+.1176`。相对参数更多的
   Raw-wide 也为 10/10 正增益。
2. **相对同结构、同可训练参数量的 Raw-PCA residual，不能声称普适胜出。** FMT 仅在
   4/10 条目、3/7 family 上取得正 F1/AP；条目平均增益为 F1 `-.0009`、AP `+.0117`，
   family-macro 为 F1 `+.0024`、AP `+.0129`。Half-cylinder 三个 Reynolds number、F-22、
   Smoke 和 Tangaroa 没有超过该强对照。

因此，对旧结论作如下公开修订：旧 `2.2+2.3` 在“选择规则显式要求 FMT 增益、且没有
Raw-PCA residual 强对照”的条件下得到 10/10；本 `3.1` 去掉该选择偏置并加入强对照后，
仍证明 FMT 明显改善原始 Raw 网络，但不证明 FMT 是所有场上都优于通用 Raw 特征扩展的
唯一表示。论文主表以本版本为准，旧结果只保留为方法开发记录。

### 可追溯文件

- 最终机器表：`outputs/mainExp_Task3_3D_3.1_ibex_v100/final_confirmation/`
- 逐 seed bootstrap 95% CI：`paper_table.csv`
- 逐时间片：`per_slice.csv`
- 选择审计：`raw_method_selection.csv` 与 `audit.json`
- 完整归档：`task3_mainExp_3.1_v100_results.tar.gz`，SHA-256
  `9aff3c0d672261e53347c7c52777ea6faad7315372052d8c2247b1e2885c2569`
- Slurm：baseline `50820277[0-1]`、residual `50829628[0-19]`、merge
  `50829629`、evaluate `50829630`。
