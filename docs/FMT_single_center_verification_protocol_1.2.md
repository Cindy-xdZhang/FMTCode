# Verify_FMT_SingleCenter_Task354C_1.2

2026-09-17用户追问为何没有部署，恢复此前统一FMT单种子核验。1.1的唯一预检52008644因部署漏掉FLowUtils而失败，正式训练未开始；失败证据保留。已补齐FLowUtils/pnn并完成远端导入检查。

本版同时遵守项目最新已确认的“原中心有效共同子集”：Task4-c只读复用`Ablation_Task4C_OriginalCenter_1.1`科学提交ad0ffc85确定的六份行号/中心索引文件，逐份SHA-256固定。训练196,042、验证2,993、测试11,288；不以邻居替换缺失中心，不修改18份原数据文件、标签或集合。与原196,960/3,000/11,320的分数分开报告。预检按原播种坐标独立复算全部六个子集，不用标签或预测筛选。

其余完整沿用[1.1协议](FMT_single_center_verification_protocol_1.1.md)：p35/n0_k06，固定一个原中心、其余全部有效邻居，141维，6频率，float64几何/傅里叶后float32输出，训练集逐特征标准化；无kin、Raw旁路、卷积或中心轮换。网络141→256→128→64→2，共78,530参数。Task3/5原十流场、标签/尺度/划分不变；三任务共同seed40，10+10+1共21次完整拟合，最多4个V100并发。训练函数和训练配置与1.1相同，新增测试比较训练函数语法树及预算字段。

本版是统一p35/n0单向量配置核验，和另一轮Task4-c原p35/h0、c156单中心及轮换对照不同；不重复提交那四个已运行作业，不挪用其结果作为本配置成绩。一个候选、一个种子不能证明所有配置中的最好。

训练前顺序为：导入检查→V100预检、18源文件及六子集哈希/几何核对、三任务各32真实训练样本工程拟合→21次正式训练→独立预测和选择轮次复算。测试读取仍在各拟合的selection.lock之后；不落盘模型权重。

实现：`experiments/FMT_SingleCenter_Task354C_1_2.py`，配置`config/Verify_FMT_SingleCenter_Task354C_1.2.json`，启动`ibex_bash/fmt_single_center_task354c_1p2.sh`，新增契约检查`tests/test_fmt_single_center_subset_1_2.py`；编码器/网络保持`FMT_Utils/FMT_SingleCenter_1_1.py`。输出`outputs/Verify_FMT_SingleCenter_Task354C_1.2/`。此恢复条目取代1.1的暂停状态，不授权其他搜索或额外种子。
