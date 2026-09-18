# mainExp_Task4C_FixedDataset_2.1：Task4-c 固定数据集 v2 的构建（实现记录）

规则的规范文本是 [Task4C_fixed_dataset_rules_v2.md](Task4C_fixed_dataset_rules_v2.md)；本文只记录实现、执行修订与冻结哈希。v2 于 2026-09-18 晚按用户要求从头重写：样本＝采样点＋三条曲线、不再有邻居线；候选区域内的初始点经 Poisson disk 降采样；归属与标签分离；以 hairpin 实例为单元分五折。v1（`mainExp_Task4C_FixedDataset_1.1`，r4 冻结）的文件与正在进行的 v1 训练（52058941–52058977）保留不动。

## 代码与配置

- 核心模块 `FMT_Utils/Task4C_FixedDataset_2_1.py`：候选区域与 26 连通分量、按体积均匀的初始点池、numba 网格哈希的贪心 dart throwing 与半径二分（`poisson_disk`）、一次积分到最长长度后按步数截取三条曲线并清洗（`trace_curves` / `cut_and_clean`）、标签与四级归属（`assign_instances`）、实例分折（`instance_folds`）、写出。
- 命令行 `experiments/Task4C_FixedDataset_2_1.py`：`verify-inputs` → `pilot`（每流场 200,000 初始点 / 1,000 样本）→ `build` → `audit-data`；`submit` 为 Ibex 链（未使用），`runtime` 记录事件。启动器 `ibex_bash/task4c_fixed_dataset_2p1.sh`。
- 配置 `config/mainExp_Task4C_FixedDataset_2.1.json`（`execution_revision` 记录修订）；原始场文件、SHA-256、λ₂ 阈值与 RK45 积分器设置读自 `config/mainExp_Task4C_GTHeadCoverage_1.1.json`，本地数据根目录 `C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D`（四个 VTK 的哈希在 `input_verification.json` 中核对通过）。
- 单元测试 `tests/test_task4c_fixed_dataset_2_1.py`（9 项），`python -m unittest tests.test_task4c_fixed_dataset_2_1` 通过。

## 执行修订

| 修订 | 变化 | 每流场初始点 / 样本 | Channel 保留 | TBL 保留 | 结果 |
|---|---|---|---:|---:|---|
| r1 | 首版 | 20,000,000 / 100,000 | 83,757 | 89,240 | 核验通过；Channel 实例 10、39 无正类（贴展向边界，曲线出域被删）；输出在 `superseded_r1/` |
| r2 | 出域半线弧长条件放宽到 [0.25L, 1.25L]（用户裁定） | 20,000,000 / 100,000 | 94,570 | 99,154 | 核验通过，全部实例有正类；输出在 `superseded_r2/` |
| **r3（当前）** | 规模改为两流场合计 4,000,000 初始点 / 400,000 样本（用户修正） | 2,000,000 / 200,000 | **189,416** | **198,413** | 核验通过 |

## r3 执行（2026-09-18 夜，本地 Windows 工作站，52 核 / 256 GB）

全部阶段在本地运行而非 Ibex：build Channel 354 s、TBL 571 s，audit-data 约 6 分钟。`preparation.json` 的 `identity` 记录 HEAD `fd87412c` 与全部源码文件哈希；**v2 代码尚未提交**，冻结前需要一次科学 commit 并把 commit 写入本文。

| 流场 | 初始点池（接受率） | Poisson 半径 r | 积分 | 保留 | hairpin / non-hairpin | 五折样本数（折 0 = 测试） | 训练 / 测试 |
|---|---|---:|---:|---:|---|---|---|
| Channel | 2,000,000（0.371） | 0.005163 | 200,000 | **189,416** | 28,878 / 160,538 | 40,670 / 37,540 / 36,957 / 37,002 / 37,247 | 148,746 / 40,670 |
| TBL | 2,000,000（0.381） | 0.10809 | 200,000 | **198,413** | 28,777 / 169,636 | 41,231 / 39,415 / 39,229 / 39,091 / 39,447 | 157,182 / 41,231 |

Channel 74 个实例、TBL 58 个实例都有正类样本。hairpin 占比约 15%。

## 核验（audit-data，通过）

`experiments/Task4C_FixedDataset_2_1.py audit-data` 从原始场重新计算：文件哈希；每个播种点三线性 λ₂ < 阈值 ∧ oyf > 0；任意两样本距离 ≥ r；以记录的种子重建初始点池并核对 `pool_index` 指向的点逐位等于保存的播种点、以记录半径重跑 dart throwing 复现选择、半径放大 10 倍容差后接受数不足 N（最大性）；三条曲线半线弧长与步数在界内、有限、无坐标轴退化、在记录的包围盒内；每流场随机 500 个样本重新积分并与保存曲线一致（atol 1e-6）；标签＝GT 单元归属；分量与四级归属重算一致；每实例只在一个折、折样本数与记录一致；`local_grid_scale` 一致。

- `outputs/mainExp_Task4C_FixedDataset_2.1/data_audit.json` SHA-256 **`e5aaa4fb8a8e01882c461da5916b3bb96883e2787b231b018b512d7e30d3a2a8`**，总样本 387,829。
- 冻结文件哈希：Channel `seeds.npy` b1ca0149…、`curves.npy` 7ea61c43…、`metadata.npz` d7301c3a…；TBL `seeds.npy` ed188f95…、`curves.npy` 8c048eda…、`metadata.npz` 5f5424f9…（完整值在 `data_audit.json` 的 `frozen_files`）。
- 数据位于 `outputs/mainExp_Task4C_FixedDataset_2.1/physical/{channel,tbl}/`；打包副本在 OneDrive `flowData3D/Task4C_fixed_dataset_v2/`（含 `docs/` 与 `SHA256SUMS.json`）及同名 zip。

## 读取约定与后续

- 中心曲线：`curves[i, k]`（k = 0/1/2 对应短/中/长），物理坐标；播种点 `seeds[i]`；邻域查询由方法自行对 `seeds` 建 KD-tree。
- 训练/测试：`metadata['split']`（0 训练 / 1 测试）按 `fold == 0`；轮换测试折只改读取方，不重建数据。
- 训练数据到 Ibex：复制 `physical/` 与 `data_audit.json` 并核对 `frozen_files` 哈希；不要在 Ibex 上重新生成。
- 尚未做：科学 commit；基于 v2 的训练/baseline 版本（需先确定各方法的邻域查询方式）。
