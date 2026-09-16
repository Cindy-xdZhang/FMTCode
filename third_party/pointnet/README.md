# PointNet attribution

Official source: https://github.com/charlesq34/pointnet

Pinned commit: `2618f72bc1a0fd21b074096e748016960d44ef55`.

MIT license, Copyright (c) 2017 Geometric Computation Group of Stanford University and Charles R. Qi; the full original notice is in `LICENSE`.

`FMT_Utils/Task4C_PointNet_1_1.py` implements the classification topology and feature-transform orthogonality penalty described in `models/pointnet_cls.py` and `models/transform_nets.py`, using PyTorch rather than TensorFlow. It keeps both identity-initialized T-Nets, shared pointwise affine/BatchNorm/ReLU layers, global max pooling, and the two-layer classifier with dropout. The channel widths are reduced to match the Task4-c FMT parameter budget, and BatchNorm and max pooling exclude padded points. See `docs/Task4C_pointnet_protocol_1.1.md` for exact widths, loss reduction, and training adaptations. No claim of reproducing the paper's original-width ModelNet40 accuracy is made.
