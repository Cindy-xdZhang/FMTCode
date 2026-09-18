# mainExp_Task4C_FixedDataset_1.1：Task4-c 固定线簇数据集 v1 的构建（实现记录）

规则的规范文本是 [Task4C_fixed_dataset_rules_v1.md](Task4C_fixed_dataset_rules_v1.md)；本文只记录实现、执行修订（r1/r2/r3）、作业与冻结哈希。本版本从头生成 Task4-c 的长期固定数据，取代 `Ablation_Task4C_FPS16_1.2` 的替换式重建（后者不满足实例覆盖规则，且核验失败于一条写错的“中心唯一”断言：模板本来允许同一中心配三种积分长度）。

## 输入与冻结参数

原始场、λ₂ 阈值（Channel −13.395、TBL −0.0272）、九种尺度（模板间距 0.25h/0.5h/1h × 三组积分参数）、RK45 双向积分、清理阈值（0.95L–1.002L 等）以及头区的 3/1/1 空间块全部读自 `config/mainExp_Task4C_GTHeadCoverage_1.1.json`；本版本不复用该数据的任何行。

## 实例划分与头区归组

每流场读取人工 GT 的全部实例（含实例 0），`instance_groups` 以固定种子（98101＋流场序号）抽取 20% 为未见组，GT 包围盒重叠的实例绑在同一组；若绑定导致恰好 20% 不可达，取不超过 20% 的最大可达数并记录。pilot 实测：Channel 74 个实例 59 覆盖 / 15 未见，TBL 58 个 46 / 12。

正类头区跟随其实例；负类头区跟随离它最近的 GT 单元所属实例。未见组头区的全部单元只进测试池；其余头区按 3/1/1 块分入训练/验证/测试池。未见实例的 GT 包围盒本身为排除区（不外扩），训练/验证中心不得落入。

## 采样、配额与生成顺序（r3）

生成顺序 **test → validation → train**，每个划分生成后立即固定洗牌。

| 划分 | 行数/流场 | GT 头部行 | 头区行 |
|---|---:|---|---|
| test | 6,000 | 覆盖组实例 `fill(('gt', i), target=10, allow_short)`；未见组实例 `fill(('box', i), ...)`：中心在 GT 包围盒内均匀随机、无任何筛选、标签＝中心 GT 归属（`sample_kind=2`）；硬性下限 2 | 余量按九种尺度均分，轮询测试池头区 |
| validation | 1,500 | 无 | 同上，验证池 |
| train | 90,000 | ① 每个覆盖实例正类测试行配对一个同实例训练中心（1h–6h 环带，须满足 GT 归属 ∧ 头部夹角 ∧ λ₂ ∧ oyf）；② 每个覆盖实例补非放松 GT 头部行到目标 10；硬性下限 2 | 余量同上，训练池 |

GT 头部行的中心从该实例满足“在 GT 内 ∧ 头部夹角 ∧ λ₂<阈值 ∧ oyf>0”的 GT 单元中心加抖动采样；头区行的中心按 4.14 规则在单元内随机播种（oyf>0 且同头区）。所有中心的 26 个模板邻居只要求在计算域内。每个来源连续 **100** 次失败后放松中心筛选（头区来源在其单元包围盒外扩一格内均匀采样，GT 来源从该实例未过滤 GT 单元采样），放松行标签取中心点 GT 归属；再 1000 次失败则该来源耗尽（配对与 GT 头部来源允许短缺，头区来源报错）。配不到训练中心的测试正类在写出前删除并重映射 `paired_test_center`。

距离规则（h 为当地网格间距几何平均）：验证中心距测试中心 ≥0.5h_test；训练中心距所有评价中心 ≥1h_eval；覆盖实例的正类测试中心距同实例训练中心 ≤6h（由配对保证）；同一划分内 (中心, 尺度) 唯一；训练/验证不入排除区。写出时再断言一次。

## 输出（每流场每划分）

`geometry.npy`、`seeds.npy`、`metadata.npz`（22 个键，与旧模板兼容，供 Conv3D/PointNet/PointNet++/BiLSTM 驱动直接读取）、`index.npz`（sample_kind、relaxed、neighbor_filter、original_center_id=0、nearest_instance、heldout_instance、center_gt_owner、center_in_gt_head、offset_test、paired_test_center）、`line_seed_attributes.npz`（seed_points、stencil_slot、exact_stencil_coordinates、lambda2、oyf、inside、head_candidate、oyf_positive、lambda2_threshold）。`preparation.json` 记录实例分组、排除区、逐实例覆盖计数、拒绝原因、放松数、删除的未配对测试行数与文件哈希，并含 `spatial_cell_and_physical_length_audit_passed` 供旧驱动检查。

## 执行链与核验

`verify-inputs`（四个 VTK 哈希）→ `pilot`（两流场，每流场 900/150/300 行、每实例目标＝下限＝2，完整走通生成与写出）→ `build`（正式配额）→ `audit-data`。核验从原始场重新计算：中心在第 0 槽、≥17 线、长度与步数、几何归一化、(中心,尺度) 唯一、逐线播种点与模板点逐位相等、λ₂/oyf/标签重插值一致、非放松中心满足 step1、GT 行标签与实例、放松行标签＝GT 归属、训练/验证不含未见实例且不进入排除区、逐实例覆盖 ≥ 下限（未见实例按包围盒内测试点计）、包围盒行只在测试且中心在盒内、1h/6h/0.5h 距离、配对行 1h–6h、测试行数＝预期−删除数。通过后 `data_audit.json` 的 30 份文件哈希即为冻结哈希。

## 执行修订记录

| 修订 | 科学 commit | 远端目录 / 作业 | 结果 | 改动 |
|---|---|---|---|---|
| r1 | ae65e920 | `FMT_Task4C_FixedDataset_1p1_20260918`；52047083–52047086 | pilot 通过；channel build 52047085_0 在 test/gt10 FAILED（训练先生成，头区内无点离训练中心 ≥3h，31,902 次 `too_close_to_train`）；其余取消 | — |
| r2 | e4500686 | `..._r2`；52048030–52048033 | pilot 52048031_0/1 FAILED：训练覆盖检查不满足（channel 11 个、TBL 4 个覆盖实例只有 1 个头部中心）；build/audit 取消，pilot 输出已删除 | 顺序改为 test→validation→train；配对训练中心；(中心,尺度) 唯一签名 |
| r3 | 见下方状态记录 | 待填 | 待运行 | 配对中心加头部夹角检查；每覆盖实例 GT 头部补到目标 10；**用户晚间规则**：评价点距训练 3h→1h、放松阈值 1000→100、覆盖下限 2；测试 GT 头部行目标 10 允许短缺至下限 2；**未见实例测试点在 GT 包围盒内随机取、无筛选，排除区不外扩** |

r2 失败根因：`pair_sample` 只要求配对中心落在 GT 包围盒内，而覆盖计数还要求头部夹角条件；落在 GT 内但不在头部区域的配对行不计入覆盖。r3 在配对时检查夹角，并在头区填充前对每个覆盖实例补 GT 头部行。

代码：`FMT_Utils/Task4C_FixedDataset_1_1.py`、`experiments/Task4C_FixedDataset_1_1.py`；配置 `config/mainExp_Task4C_FixedDataset_1.1.json`（`execution_revision` 记录修订）；启动器 `ibex_bash/task4c_fixed_dataset_1p1.sh`；测试 `tests/test_task4c_fixed_dataset_1_1.py`（实例分组、距离规则、配对环带与夹角、补样顺序、放松、耗尽、写出、配置，共 8 项）。训练与 baseline 比较在数据冻结后以 `mainExp_Task4C_FixedDatasetFMT_1.1` 与 `mainExp_Task4C_FixedDatasetBaselines_1.1` 进行。
