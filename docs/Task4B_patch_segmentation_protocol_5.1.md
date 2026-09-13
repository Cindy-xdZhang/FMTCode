# mainExp_Task4B_PatchSegmentation_5.1

2026-09-06预注册；用户要求channel+TBL各900个训练区域、100个测试区域，hairpin实例9:1，共享FMT四分类分割网络。按已说明的空间隔离方案构建，无模型结果参与划分。旧版代码、标签与指标保持不变。

## 标签与划分

沿用4.4完整网格代理标签：Channel IVD>5.785，TBL IVD>0.08。四类为ordinary_streamwise、ordinary_spanwise、hairpin_head、hairpin_leg。hairpin成员×velocity-curl角度二分，45度归平行、反平行同属平行。非涡区ignore=-1，四分类只在IVD涡区进行，不宣称网络预测了涡/非涡边界。

每个原始instance在冻结GT体素网格上的包围盒向各轴扩张1体素、在全域边界裁剪，构成一个hairpin局部区域。盒子距离不足1体素的实例连成组，同组全部归于同一集合。固定随机序列的整数子集选择，得到Channel 66/7、TBL 52/6个训练/测试实例，不按类别性能或训练结果挑选。几何组约束意味着实例归属不是逐实例独立随机抽签。包围盒在IVD过滤前计算；核对体素instance集合等于原始GT instance集合。

随机区域大小从训练实例包围盒的尺寸列表抽取，在原流场内均匀随机滑动。先补足100个测试区域（拒绝与训练实例盒相交或间隔不足1体素的窗口），再补足900个训练区域（拒绝与任何测试区域相交或间隔不足1体素的窗口）。不按四类比例或模型表现接受窗口；允许无涡区窗口。集合内部窗口可重叠，跨集合窗口及采样体素不重叠。每个窗口不得含另一个集合的任何hairpin实例。完整manifest记录2000个区域的z/y/x半开网格边界、实例、类型、split。

## 区域内primitive与网络

primitive指每个采样点附近的7条流线簇，中心线及沿三轴正负偏移的6条线。每条双向各16步、33点，四阶Runge-Kutta单位速度积分。所有查询及其原生速度插值节点必须位于当前patch内部；Channel不对小区域人为施加周期条件，去除全场x/y的重复周期端点以避免跨集合重复读取同一原生节点。

每个seed的物理步长h=min(原4.1的0.5×全场目标网格平均间距, 0.95×seed到patch内原生插值域最近边界距离/(16+offset/h+1))。offset/h=0.1501458034894056。这是本版为小区域新增的边界约束，并非沿用4.1完整场的固定物理步长。积分在以seed为原点、h为单位的局部坐标进行，物理查询float64；FMT输入再减去中心线首点。每flow速度均方根仅用于数值低速有效性检查，单位速度积分不受速度整体倍率影响。步长、查询空间范围及无效点数全部记录；不声称不同flow或不同patch尺度完全等价。

训练patch保留全部hairpin涡区点，ordinary涡区每patch均匀无放回最多256点；测试patch全部涡区体素逐点推理。无涡区patch计入区域数，不进入四分类损失。无效primitive保留候选记录并报告，预测为-2；不得悄悄从覆盖分母删除。

FMT为固定161维：6频、Gram、chirality、sorted neighbors，neighbor_scale=100；均值与标准差仅由两个flow合并的有效训练样本计算，标准化后neighbor block乘0.5。网络沿用PathlineMulticlassClassifier3D的fmt_only结构161→1024→1024→4、LayerNorm/GELU，本版dropout=0.1。每点预测再放回patch网格形成分割；模型不接收坐标、instance ID、flow ID、速度、curl或IVD作为输入，也没有新增patch间上下文网络。

单个共享模型，seed8073；Adam lr1e-3、weight_decay1e-4，余弦退火到1e-6，固定200epoch、batch1024，每epoch训练样本无放回一遍。交叉熵类别权重为合并有效训练类别频数的平方根倒数，再除以权重均值。训练轮数固定，不设额外validation，不用test选epoch、参数或模型。仅训练结束后一次最终test推理；不保留checkpoint。

## 输出和证据边界

报告每flow的有效点指标及含无效点惩罚的accuracy、四类F1、IoU（Intersection over Union，预测与参考区域交集除以并集），以及mIoU（四类IoU的平均）。无效预测计为对应真实类的漏检。按patch出现次数的指标与按全场唯一体素合并的指标分开；同集合重叠patch对同一体素的概率取平均，主结果用去重后的test体素。每patch四类缺席时该类IoU记0，patch平均仅对含涡点patch求均值；完整200个test patch数量与空patch数另报。

导出全部测试区域的体素索引、类别、预测、概率及分割数组；三维对照固定选每flow最先两个测试hairpin包围盒，不根据预测好坏挑图。结果只支持这两个已知flow内未见空间区域/实例的性能，不构成跨flow泛化或FMT优于Raw的证据。旧4.1训练样本记忆结果不能替代本版测试指标。

代码：FMT_Utils/Task4B_PatchSplit_3D.py；experiments/Task4B_PatchSegmentation_5_1.py；config/mainExp_Task4B_PatchSegmentation_5.1.json；tests/test_task4b_patch_segmentation_5_1.py。运行记录见ibex_run_registry，方法结果只记experiment_log。
