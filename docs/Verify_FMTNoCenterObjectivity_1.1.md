# Verify_FMTNoCenterObjectivity_1.1

目标：直接验证用户所指第三种方案，即原冻结FMT只将中心23维置零，邻居138维原样保留。额外审计原性能实验的28维kin4辅助块；核心与辅助块必须分开判断。方法级结论只写入 `docs/experiment_log.md`。

代码：`experiments/Verify_FMTNoCenterObjectivity_1_1.py`。显式复用冻结编码器的6频率、邻居缩放1、权重1、排序池化、频域范数/内积及手性配方，唯一干预是中心23维置零；逐元素断言其他维度未变，执行前后核对冻结编码器文件哈希。

输入：轮廓系数项目全部8个真实几何文件的所有七线primitive，32采样点，文件路径及SHA256逐项保存。另设中心及六个轴向单位邻居全部静止的解析构造，满足同一材料点连续轨迹定义。无标签、无训练、无新播种。

预定义变换：恒等、时变平移、恒定90°旋转、时变旋转、时变旋转加平移。旋转为z轴0至1.3弧度随采样时间线性变化；平移为(t,t²,sin(t))，t为0到1的窗口相位。每个时间所有材料点共享同一Q(t)与c(t)，断言正交性与行列式为1。几何对照检查全部21对同时间距离、同时间内积矩阵以及相对向量的正确变换关系。

所有输入分别以float64和float32执行；float64用于区分结构性变化与float32舍入。恒等流程必须逐位相同。报告原始单位最大绝对误差、L2范数误差、对称相对L2误差及超过rtol=atol=1e-5的primitive数量，不跨表示用绝对误差作性能排名。单个合法变换下稳定的反例即可否定任意观察者不变性；有限测试通过本身不证明普遍不变。

```powershell
./tmp/jhtdb-venv/Scripts/python.exe -m experiments.Verify_FMTNoCenterObjectivity_1_1
```

输出目录 `outputs/Verify_FMTNoCenterObjectivity_1.1` 拒绝覆盖，包含结果JSON、逐块误差CSV和静止七点反例的完整输入/输出NPZ。`execution_status`指验证程序是否完成，`encoder_observer_invariance`单列编码器判定，避免将测试执行成功误读为算法客观。只在本地运行，不提交Task3/Task5训练，不修改冻结算法或历史指标。
