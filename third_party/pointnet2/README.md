# PointNet++ attribution

Primary source: Charles R. Qi et al., *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space*, NeurIPS 2017, https://arxiv.org/abs/1706.02413.

Official implementation: https://github.com/charlesq34/pointnet2/tree/42926632a3c33461aebfbee2d829098b30a23aaa, MIT license preserved in `LICENSE`.

`FMT_Utils/Task4C_PointNetPlusPlus_1_1.py` implements the official single-scale classification topology in native PyTorch. Sources consulted: `models/pointnet2_cls_ssg.py`, `utils/pointnet_util.py`, and `tf_ops/grouping/tf_grouping_g.cu` at the pinned commit. The grouping algorithm retains the first points strictly within the ball and repeats its first point if underfull.

Task4-c adaptations: binary output, reduced channel widths for matched learned parameters, variable valid-point counts with padding excluded, deterministic canonical xyz ordering and double-precision sampling/grouping arithmetic, and cached geometry-only indices. Training uses the frozen Task4-c settings rather than the paper's ModelNet40 augmentation/training recipe. This is not a reproduction of the reported ModelNet40 score. Complete details are in `docs/Task4C_pointnetplusplus_protocol_1.1.md`.
