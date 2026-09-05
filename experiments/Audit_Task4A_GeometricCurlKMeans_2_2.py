"""Independent result audit for Verify_Task4A_GeometricCurlKMeans_2.2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_RESULT = Path(
    "outputs/Verify_Task4A_GeometricCurlKMeans_2.2/channel_GTs/"
    "geometric_curl_kmeans_result.npz"
)
DEFAULT_SUMMARY = DEFAULT_RESULT.with_name("summary.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_f1(truth: np.ndarray, prediction: np.ndarray, label: int) -> float:
    truth_mask = truth == label
    prediction_mask = prediction == label
    true_positive = int(np.count_nonzero(truth_mask & prediction_mask))
    false_positive = int(np.count_nonzero(~truth_mask & prediction_mask))
    false_negative = int(np.count_nonzero(truth_mask & ~prediction_mask))
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else 2.0 * true_positive / denominator


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()
    result_path = Path(args.result)
    summary_path = Path(args.summary)
    result = np.load(result_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    ids = np.asarray(result["vortex_ids"], dtype=np.int32)
    prediction = np.asarray(result["geometry_kmeans_classes"], dtype=np.int8)
    proxy = np.asarray(result["proxy_classes"], dtype=np.int8)
    if not (len(ids) == len(prediction) == len(proxy) == 16711):
        raise AssertionError("expected exactly 16,711 aligned positive cubes")
    unique_ids = np.unique(ids)
    if len(unique_ids) != 73 or unique_ids.min() <= 0:
        raise AssertionError("expected exactly 73 positive VortexIds")
    if np.any(~np.isin(prediction, (1, 2))) or np.any(~np.isin(proxy, (1, 2))):
        raise AssertionError("prediction and proxy must assign every cube")

    per_id = []
    for vortex_id in unique_ids:
        member = ids == vortex_id
        if set(np.unique(prediction[member])) != {1, 2}:
            raise AssertionError(f"VortexId {int(vortex_id)} lacks one KMeans class")
        stream_f1 = _class_f1(proxy[member], prediction[member], 1)
        span_f1 = _class_f1(proxy[member], prediction[member], 2)
        per_id.append(
            {
                "vortex_id": int(vortex_id),
                "cube_count": int(np.count_nonzero(member)),
                "macro_f1": 0.5 * (stream_f1 + span_f1),
                "hard_label_mse": float(np.mean(prediction[member] != proxy[member])),
            }
        )
    macro_f1 = float(np.mean([row["macro_f1"] for row in per_id]))
    hard_mse = float(np.mean([row["hard_label_mse"] for row in per_id]))
    official = summary["final_geometry_named_kmeans_metrics"]
    official_f1 = float(official["primary_vortex_equal_weighted_macro_f1"]["mean"])
    official_mse = float(official["vortex_equal_weighted_hard_label_mse"]["mean"])
    tolerance = 1e-12
    if abs(macro_f1 - official_f1) > tolerance:
        raise AssertionError("independent macro-F1 does not match summary")
    if abs(hard_mse - official_mse) > tolerance:
        raise AssertionError("independent hard MSE does not match summary")

    audit = {
        "experiment": summary["experiment"],
        "result_sha256": _sha256(result_path),
        "summary_sha256": _sha256(summary_path),
        "cube_count": len(ids),
        "vortex_id_count": len(unique_ids),
        "all_ids_have_both_kmeans_classes": True,
        "independent_vortex_equal_weighted_macro_f1": macro_f1,
        "independent_vortex_equal_weighted_hard_label_mse": hard_mse,
        "max_absolute_difference_from_summary": max(
            abs(macro_f1 - official_f1), abs(hard_mse - official_mse)
        ),
        "ids_macro_f1_at_least_0_90": int(
            sum(row["macro_f1"] >= 0.90 for row in per_id)
        ),
        "ids_hard_mse_at_most_0_10": int(
            sum(row["hard_label_mse"] <= 0.10 for row in per_id)
        ),
        "success_gate_recomputed": bool(macro_f1 >= 0.90 and hard_mse <= 0.10),
    }
    output_path = result_path.with_name("independent_audit.json")
    audit["audit_path"] = str(output_path.resolve())
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


if __name__ == "__main__":
    main()
