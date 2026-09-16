"""Traceable, additive explanations of a fixed two-cluster feature partition."""
from __future__ import annotations

import numpy as np


BLOCKS = [
    dict(id="center", name="中心步进频谱", start=0, stop=23),
    dict(id="direction", name="坐标／切向方向频谱", start=23, stop=95),
    dict(id="neighbor_mean", name="邻居频谱逐特征均值", start=95, stop=118),
    dict(id="neighbor_max", name="邻居频谱逐特征最大值", start=118, stop=141),
]


def p35_schema():
    """Match primitive_features concatenation, including the zero DC slots."""
    result = []
    for block in BLOCKS:
        for index in range(block["start"], block["stop"]):
            j = index - block["start"]
            item = dict(index=index, block=block["id"], block_name=block["name"])
            if block["id"] == "direction":
                k, rem = divmod(j, 12)
                channel, part = divmod(rem, 2)
                ch = ["q_x", "q_y", "q_z", "t_x", "t_y", "t_z"][channel]
                op = ["Re", "Im"][part]
                item.update(frequency=k, channel=ch, component=op,
                    name=f"方向 · {ch} · k={k} · {op}",
                    formula=f"{op}(rFFT_ortho({ch})[{k}])",
                    sequence="q为整组质心/最大半径归一化后的中心线坐标；t为相邻点单位切向，末点复制前一切向。序列32点。",
                    theoretical_zero=(k == 0 and part == 1))
            else:
                if j < 18:
                    k, kind = divmod(j, 3)
                    term = ["实部范数", "虚部范数", "实虚夹角余弦"][kind]
                    formula = [f"||a_{k}||", f"||b_{k}||",
                        f"(a_{k}·b_{k}) / max(||a_{k}|| ||b_{k}||, 1e-8)"][kind]
                    zero = k == 0 and kind in (1, 2)
                else:
                    k = j - 18
                    term = f"有向三重积 k={k}→{k+1}"
                    formula = f"((a_{k}×b_{k})·a_{k+1}) / max(||a_{k}|| ||b_{k}|| ||a_{k+1}||, 1e-8)"
                    zero = k == 0
                pool = {"center": "中心", "neighbor_mean": "邻居均值", "neighbor_max": "邻居最大值"}[block["id"]]
                sequence = ("u[n]=q0[n+1]−q0[n]，31个中心步进向量。" if block["id"] == "center" else
                    "u_i[n]=(qi[n+1]−q0[n+1])−(qi[n]−q0[n])，31个邻居相对变化向量；倍率为1。")
                if block["id"] != "center":
                    formula = ("mean_i(" if block["id"] == "neighbor_mean" else "max_i(") + formula + ")"
                    sequence += "先各自编码六个邻居，再逐特征取算术均值或数值最大值；不取绝对值最大值。"
                item.update(frequency=k, component=term,
                    name=f"{pool} · k={k} · {term}", formula=formula,
                    sequence=sequence + " F[k]=a_k+i b_k，rFFT不除以序列长度。",
                    theoretical_zero=zero)
            item["label"] = f"[{index}] {item['name']}"
            result.append(item)
    return result


def explain_partition(raw, standardized, labels, centers):
    """All arrays are evaluated in float64; no ground-truth labels are used."""
    raw, z = np.asarray(raw, dtype=np.float64), np.asarray(standardized, dtype=np.float64)
    labels, centers = np.asarray(labels), np.asarray(centers, dtype=np.float64)
    if raw.shape != z.shape or z.ndim != 2 or centers.shape != (2, z.shape[1]):
        raise ValueError("Expected matching feature matrices and two centers")
    if not np.isfinite(raw).all() or not np.isfinite(z).all() or not np.isfinite(centers).all():
        raise ValueError("Nonfinite features")
    if labels.shape != (len(z),) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both clusters must be nonempty")
    distance = ((z[:, None, :] - centers[None]) ** 2).sum(-1)
    if not np.array_equal(distance.argmin(1), labels):
        raise ValueError("Labels are not the nearest-center assignments")
    means = np.stack([z[labels == c].mean(0) for c in (0, 1)])
    if not np.allclose(means, centers, atol=1e-8, rtol=1e-8):
        raise ValueError("Centers do not equal cluster means; fit has not converged")
    delta = means[1] - means[0]
    square = delta ** 2
    if square.sum() <= 0:
        raise ValueError("Identical cluster centers have no contrast")
    contribution = (z - centers[1]) ** 2 - (z - centers[0]) ** 2
    margin = distance[:, 1] - distance[:, 0]
    error = np.abs(contribution.sum(1) - margin).max()
    if not np.allclose(contribution.sum(1), margin, rtol=1e-11, atol=1e-10):
        raise ValueError("Per-dimension distance decomposition failed")
    return dict(mean=means, raw_mean=np.stack([raw[labels == c].mean(0) for c in (0, 1)]),
        std=np.stack([z[labels == c].std(0) for c in (0, 1)]),
        raw_std=np.stack([raw[labels == c].std(0) for c in (0, 1)]),
        delta=delta, square=square, share=square / square.sum(),
        ranking=np.argsort(-square, kind="stable"), contribution=contribution,
        margin=margin, distance=distance, decomposition_error=float(error),
        counts=np.bincount(labels, minlength=2))
