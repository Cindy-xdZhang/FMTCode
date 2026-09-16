# Point-NN attribution

Upstream: https://github.com/ZrrSkywalker/Point-NN

Pinned commit: `a85bfc365258a2c65a5f6ac289537b4d7a7cec0f`.

Author: Renrui Zhang. License: MIT; the complete original notice is in `LICENSE`.

`FMT_Utils/Task6PNNTrans_3D.py` adapts `PosE_Initial`, `PosE_Geo`, `LGA`, and the evaluation-time pooling equations from upstream `models/point_nn.py`. It uses native PyTorch on seven material points, keeps every anchor identity, normalizes within each snapshot rather than across the batch, and has no learnable affine normalization. The temporal Transformer and reconstruction decoder are new project code. No upstream custom CUDA extension or label memory is imported. Full design differences are recorded in `docs/Task6_pnn_trans_protocol_1.1.md`.

`FMT_Utils/Task4C_GeometricBaselines_1_1.py` separately adapts the whole-cloud four-stage encoder for all valid points in a vortex-line bundle. It preserves the upstream default widths (72 to 1152), maximum 90 neighbors, sine/cosine equations, and local/global max-plus-mean pooling. Native deterministic farthest-point sampling and lexicographic point ordering replace the custom CUDA extension; normalization is per cloud; fixed evaluation-mode BatchNorm is expressed without parameters. Variable valid cloud sizes halve at each stage, with the neighbor count capped by the available points. A newly trained MLP replaces the paper's label memory classifier, as explicitly requested by the user. See `docs/Task4C_geometric_baselines_protocol_1.1.md`. The frozen Task6 implementation is unchanged.
