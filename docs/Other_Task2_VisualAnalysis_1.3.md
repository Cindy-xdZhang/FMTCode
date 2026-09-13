# Task2 visual analysis 1.3：自由起始时间、末尾截断、UMAP和KMeans

版本 `Other_Task2_VisualAnalysis_1.3`。用户明确要求取消交互重算的数据划分限制；1.1/1.2的历史结果不改写。

## 使用

选择流场、起始时间t0、步长dt、步数N，点击重新计算。t0可位于两个源帧之间，通过速度插值得到起始场。实际终点为 `min(t0+N*dt, source_tmax)`，不再受到confirmation起始时间限制。界面显示源时间范围、请求终点、实际终点与时长。t0必须在源数据范围内且早于末尾；Cylinder默认起点仍在后50%时间，但用户可明确输入源数据范围内的其他时间。

源数据均匀时钟使用所载窗口的实际首末时间建立，不累积float32首帧间隔误差。使用请求dt积分完整步，最后剩余时间用一个短的四阶Runge–Kutta步完成。到末尾后的时间不外推。每条轨线按实际积分时间均匀线性重采样为32点；即使剩余时间不足31个dt也可显示。保持七线primitive，空间出界或无效的primitive仍会剔除。

每次只生成用户指定t0的一批轨线，用全部有效primitive拟合两臂VAE并编码同一批样本，不另划训练/显示/测试集合。此为样本内几何探索，不报告独立测试性能。两臂VAE仍使用冻结架构、seed100和各7000步；所有主表实验保持原协议。

降维方法可选t-SNE或UMAP；聚类方法可选DBSCAN或KMeans；聚类空间可选当前二维投影或高维VAE潜在向量。UMAP（Uniform Manifold Approximation and Projection，统一流形近似与投影）提供n_neighbors和min_dist控件，默认15和0.1，二维、欧氏距离、seed7068、单线程。参数含义依据[UMAP官方文档](https://umap-learn.readthedocs.io/en/latest/parameters.html)。KMeans提供簇数K，默认8，n_init=10、seed7068；所有样本均分配类别，不定义噪声或核心点。仅启用所选方法的相关控件，点击Apply clustering应用。

降维方式改变时分别缓存投影。若在高维聚类，改变投影不改变相同参数下的簇标签。二维投影改变距离和密度，DBSCAN的eps不能在t-SNE/UMAP之间直接等同。隐藏簇仍同时作用于两视图。

## 代码和记录

- `FMT_Utils/Task2GeometryRecompute.py`：无划分的重积分、短步截断、重采样和同批样本拟合。
- `FMT_Utils/Task2VisualAnalysis.py`：真实umap-learn投影、sklearn KMeans聚类及图形/导出。
- `experiments/Task2_Visual_Analysis.py`：交互控件、方法参数、结果缓存和导出。
- `config/Other_Task2_VisualAnalysis_1.3.json`：新版本配置，默认入口。
- `requirements_task2_visual_analysis.txt`：附加交互依赖；在现有FMT运行环境安装。

每次重算独立目录保存job.json、geometry_slice_00.npz、latent_bundle.npz/json和两臂默认报告；记录t0、请求/实际终点、是否截断、完整dt步数、最后短步、源范围、原始种子ID、逐臂损失和代码/数据哈希。不保存模型checkpoint。

导出记录实际projection和cluster_method，UMAP坐标命名为umap/umap_1/umap_2而非tsne；KMeans的CSV核心点列为空，JSON标记core_status_applicable=false。NPZ中的is_core为全False占位，以该标记解释。

真实验证与历史说明见docs/experiment_log.md。
