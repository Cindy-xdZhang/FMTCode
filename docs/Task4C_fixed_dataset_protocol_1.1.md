# mainExp_Task4C_FixedDataset_1.1：Task4-c 固定线簇数据集 v1 的构建

本版本实现总协议 2b 节的全部规则（样本＝采样点、只筛中心、≥17 线、放松按 GT 归属、hairpin 实例覆盖与实例划分、逐线播种点与标签），从头生成 Task4-c 的长期固定数据。它取代 `Ablation_Task4C_FPS16_1.2` 的替换式重建（后者不满足实例覆盖规则，且核验失败于一条写错的“中心唯一”断言：模板本来允许同一中心配三种积分长度）。

## 输入与冻结参数

原始场、λ₂ 阈值（Channel −13.395、TBL −0.0272）、九种尺度（模板间距 0.25h/0.5h/1h × 三组积分参数）、RK45 双向积分、清理阈值（0.95L–1.002L 等）以及头区的 3/1/1 空间块全部读自 `config/mainExp_Task4C_GTHeadCoverage_1.1.json`；本版本不复用该数据的任何行。

## 实例划分

每流场读取人工 GT 的全部实例（含实例 0），`instance_groups` 以固定种子（98101＋流场序号）抽取 20% 为未见组，GT 包围盒重叠的实例绑在同一组；若绑定导致恰好 20% 不可达，取不超过 20% 的最大可达数并记录。Channel 74 个实例目标 15 个未见，TBL 58 个目标 12 个。

头区归组：正类头区跟随其实例；负类头区跟随离它最近的 GT 单元所属实例。未见组头区的全部单元只进测试池；其余头区按 3/1/1 块分入训练/验证/测试池。未见实例的 GT 包围盒外扩一个中值网格间距为排除区，训练/验证中心不得落入。

## 采样与配额（每流场）

| 划分 | 行数 | GT 头部覆盖行 | 头区行 |
|---|---:|---|---|
| train | 90,000 | 每个覆盖实例 ≥10 | 余量按九种尺度均分，均匀轮询训练池头区 |
| validation | 1,500 | 无 | 同上，验证池 |
| test | 6,000 | **每个实例** ≥10（覆盖实例为偏移点，未见实例为新点） | 同上，测试池（含未见组头区全部单元） |

GT 头部覆盖行：中心从该实例满足“在 GT 内 ∧ 头部夹角 ∧ λ₂<阈值 ∧ oyf>0”的 GT 单元中心加抖动采样；头区行：中心按 4.14 规则在单元内随机播种（oyf>0 且同头区）。两类中心的 26 个模板邻居只要求在计算域内。每个来源连续 1000 次失败后放松中心筛选（头区行在其单元包围盒外扩一格内均匀采样，GT 行从该实例未过滤 GT 单元采样），放松行标签取中心点 GT 归属，再 1000 次失败则报错。

距离规则（h 为当地网格间距几何平均）：验证/测试中心距任意训练中心 ≥3h；覆盖实例的正类测试中心距同实例训练中心 ≤6h；测试中心距验证中心 ≥0.5h；同一划分内 (中心, 尺度) 唯一。生成顺序 train → validation → test。

## 输出（每流场每划分）

`geometry.npy`、`seeds.npy`、`metadata.npz`（22 个键，与旧模板兼容，供 Conv3D/PointNet/PointNet++/BiLSTM 驱动直接读取）、`index.npz`（sample_kind、relaxed、neighbor_filter、original_center_id=0、nearest_instance、heldout_instance、center_gt_owner、center_in_gt_head、offset_test）、`line_seed_attributes.npz`（seed_points、stencil_slot、lambda2、oyf、inside、head_candidate、oyf_positive）。`preparation.json` 记录实例分组、排除区、逐实例覆盖计数、拒绝原因、放松数与文件哈希，并含 `spatial_cell_and_physical_length_audit_passed` 供旧驱动检查。

## 执行链与核验

`verify-inputs`（四个 VTK 哈希）→ `pilot`（两流场，每流场 900/150/300 行、每实例 ≥2 覆盖，完整走通生成与写出）→ `build`（正式配额）→ `audit-data`。核验从原始场重新计算：中心在第 0 槽、≥17 线、长度与步数、几何归一化、(中心,尺度) 唯一、逐线播种点与模板点逐位相等、λ₂/oyf/标签重插值一致、非放松中心满足 step1、GT 行标签与实例、放松行标签＝GT 归属、训练/验证不含未见实例且不进入排除区、逐实例覆盖 ≥ 最低值、3h/6h/0.5h 距离。通过后 `data_audit.json` 的 30 份文件哈希即为冻结哈希。

代码：`FMT_Utils/Task4C_FixedDataset_1_1.py`、`experiments/Task4C_FixedDataset_1_1.py`；配置 `config/mainExp_Task4C_FixedDataset_1.1.json`；启动器 `ibex_bash/task4c_fixed_dataset_1p1.sh`；测试 `tests/test_task4c_fixed_dataset_1_1.py`（实例分组、距离规则、放松、写出、配置）。训练与 baseline 比较在数据冻结后以独立版本进行。
