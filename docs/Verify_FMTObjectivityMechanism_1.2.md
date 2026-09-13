# Verify_FMTObjectivityMechanism_1.2

2026-09-09，1.1集群预检失败后建立修订版本；仍遵循1.1的12臂、10数据条目、3种子、晚期Cylinder划分、700维容量控制和全部评估协议。

原因：新诊断误用了`pathline_dft_features_3d`函数默认`neighbor_scale=100, neighbor_weight=.5`，但冻结缓存生成器`Build_Task2_Universality_Cache.py:135`明确使用二者均1.0。由于手性三重积有分母下限，缩放不一定能被最终StandardScaler完全抵消。1.1预检在第一个Channel输入处以原定rtol/atol2e-4拒绝缓存复放；未产生正式Task1/2测试指标。1.1的3更新smoke已完成，不能作为有效科学对照。

修订：显式以缓存生成参数1.0/1.0重算旧特征；“只去掉邻居时间差分”臂也使用相同1.0/1.0，使之真正只改变差分操作。原冻结FMT函数、缓存和已有主表均不修改。数值容差保持不变；增加缓存参数单元测试，总7项。运行入口代码路径复用，1.1原始7文件上传包及远端独立目录完整保留，以源码SHA区分版本。

1.2在新的`/home/zhanx0o/FMT_ObjectivityMechanism_20260909_1p2`运行，输出独立`Verify_FMTObjectivityMechanism_1.2`目录。原1.1失败预检和已完成smoke保留；未启动的依赖数组及其审计取消并登记。全部1.2任务重新依赖新预检和新smoke，测试前仍冻结所有候选，未读取任何正式测试性能用于本修订。
