# Task6：逐帧 observed field 与 vortex coreline 标签

**Task6 的基本数据单元是：一个流场的一个时刻的 observed 速度场 v−u，以及该帧的 coreline 曲线标签（包括人工修订结果及待人工检查的自动结果）。** 稳态 tornado 直接使用原速度 v。流线可以按需求重新积分；本包不保存固定种子、预积分流线、点分类标签或邻居表。

当前帧列表、物理时间和核线数量以 `dataset.json` 为准；训练、测试角色以 `default_split.json` 为准。原69帧的人工标签和90项人工操作完整保留；2026-09-22另加入四场各8个非空自动核线帧，作为训练帧供人工检查。目前共101帧（77训练、24测试）：Re160、Re320、DeltaWing各23帧，Re64015帧，SquareCylinder16帧，稳态tornado1帧。此前7个SquareCylinder排除帧保持排除。

## 文件与坐标

每个 `frames/<flow>/frame_###/` 包含：

| 文件 | 含义 |
|---|---|
| `relative_velocity.npy` | float32，形状 `[Nz,Ny,Nx,3]`，最后三项依次为物理速度x/y/z，未归一化 |
| `coordinates.npz` | 一维物理坐标 `x`、`y`、`z`，不按轴缩放 |
| `corelines.vtp` | 可在 ParaView 等软件读取的人工核线标签 |
| `core_points.npy`、`core_offsets.npy` | 同一批核线的纯NumPy读取形式；第i条为 `points[offsets[i]:offsets[i+1]]` |
| `ivd.npy`、`ivd.json` | 辅助标量：由原速度v计算的瞬时涡量偏差（IVD），用于需要时复现候选区；不是由v−u计算 |
| `observer_provenance.json` | 原观察者计算参数、所用原速度切片及程序的哈希 |

每帧目录仅包含表中8个文件。物理时间、范围、网格、h、稳定核线ID及文件哈希统一保存在根目录 `dataset.json`，不在每帧重复保存 `frame.json`。

所有核线和速度场使用同一物理坐标。`h=min((xmax−xmin)/(Nx−1),...)`。IVD候选区为原v的IVD严格大于该帧全网格算术均值；这是旧采样协议，不限制合作者重新选择种子。若复现旧点分类任务，在任意查询点到连续核线线段的距离严格小于h时记为正类。

## 读取与按需积分（只返回内存）

```bash
python -m pip install -r requirements.txt
python task6_fields.py verify
```

```python
import numpy as np
from task6_fields import Dataset

ds = Dataset('.')
frame = ds.frame('cylinder3d:81')
v_minus_u = frame.velocity       # 只读内存映射，完整场
cores = frame.corelines()        # 人工核线标签
rng = np.random.default_rng(42)
seeds = rng.uniform(frame.bounds[:, 0], frame.bounds[:, 1], (1000, 3))
result = frame.integrate(seeds, total_length=1.0, points=129)
curves = result['curves'][result['valid']]
```

更换随机数种子、数量、物理总弧长即可重新积分，不重算GGT17。积分器双向积分，各方向目标为总长的一半，边界/停滞处截断；h/4与h/8两次积分检查最大差≤h/20，须使用返回的 `valid` 掩码，不能把失败轨迹当正常样本。不自动写入曲线文件，不产生新的固定曲线数据集。数组中的速度是速度值；积分器使用归一化方向来控制物理弧长。

## 帧级训练/测试划分

`default_split.json` 仅列出 `flow:index`，数据本身与训练/测试角色分开。复制该JSON并修改train/test/unused列表，再用 `Dataset('.', split='my_split.json')` 读取；同一帧不可进入两个集合，每帧必须在某一列表中。可按比例或物理时间任意划分，无需重建速度场和标签。`dataset_sha256` 固定对应数据快照，修改划分时保留它。

## 从原流场生成新的 observed 帧

详见 [GGT17 调用说明](GGT17.md)。Windows x64已附实际生成本数据的 `tools/bin/reference.dll`；脚本直接调用该C++库，不使用另一套Python近似求解器。例如：

```bash
python tools/ggt17_observe.py --flow cylinder3d --source "D:/flowdata/halfcylinderRe160.nc" --index 81 --out "D:/new_frames/re160_081"
python tools/ggt17_observe.py --flow SquareCylinder --source "D:/flowdata/SquareCylinderAmira" --index 65 --out "D:/new_frames/square_065"
```

索引从0开始。输出新场和生成记录，不会自动伪造新帧的人工核线标签；新的核线仍需要提取和检查。原始全时间序列不重复打包，使用已有原流场即可。包内已有速度场的帧无需重新运行此脚本。

## 发布与历史

8872核线编辑直接使用本目录。编辑操作自动保存到 `annotations/coreline/draft.json`；点击“应用全部修改到数据集”后更新本目录的核线文件、根清单和 `current.json`。历史标签修订保存在 `annotations/coreline/revisions`，迁移前人工记录位于 `provenance/manual_history_before_field_editor.zip`；不得用自动标签覆盖人工修改。IVD阈值沿用原编辑器的draft存储，与核线发布独立。旧预积分样本包不是Task6的本质数据。以前共享说明中的“完整数据集”只包含训练样本与标签，缺少场，此包修正了这个问题。原Ibex训练的冻结输入和已提交实验不会因本机重新打包而被偷偷修改。
