# Verify_Task3_BalancedCombination_14.1

## 问题

在不重复完整五因素全因子搜索的前提下，检验冻结的 class-balanced
mini-batch 配方能否进一步改善 11.1 四因素组合赢家的
`FMT residual − same-width Raw-PCA residual` 增益。

## 冻结协议

- 本实验只读取原 development population，不读取 12.1 confirmation。
- preflight 同时等待并冻结 11.1 与 13.1 两个完整 selector 的 SHA-256、
  family-specific recipe 和 paired seeds 40–42。
- 两个预声明候选只有 `11.1 core` 与 `11.1 core + 13.1 balanced`；配置在
  两个源 selector 完成前建立，不能根据它们的结果改候选。
- FMT 与 Raw-PCA 两臂共用完全相同的组合训练配方、batch索引、模型结构、
  参数预算与随机种子。
- 任何 family 中重复定义且取值不同的训练超参数都会使 preflight 明确失败，
  不会静默覆盖。

## 规模和判据

- 2 candidates × 10 datasets × 3 seeds × 2 arms = 120 次训练。
- 主排序：development dataset-macro paired F1 gain。
- 依次用 paired Average Precision gain、正增益数据集数、最差数据集 F1 gain
  和最差 seed F1 gain打破平局。
- 开发目标为 `>= +0.165`；未达到也必须保留。
- 本实验是development搜索。若产生新赢家，必须使用未见空间population另做
  confirmation，不能把12.1或14.1开发数据冒充最终结果。
