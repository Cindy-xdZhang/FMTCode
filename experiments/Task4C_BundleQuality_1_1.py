"""Channel-only Task4-c: inspect, prepare/review bundles, or compare models.

Four-class training requires explicit human labels and disjoint bundle groups.
All model weights remain in memory; results never include checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import socket
import subprocess
import time

import numpy as np

from FMT_Utils.Task4C_Bundles_1_1 import (
    AuxiliaryEncoder, fmt_bundle_features, normalize_bundle, read_vtk, trace_vortex_lines,
)
from FMT_Utils.Task4C_PaperBaseline_1_1 import (
    CLASS_NAMES, PaperAutoencoder, AsymmetricMarginClassifier,
    inverse_frequency_weights, reconstruction_loss, structural_confidence,
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def identity(config):
    sources = [__file__, "FMT_Utils/Task4C_PaperBaseline_1_1.py", "FMT_Utils/Task4C_Bundles_1_1.py",
               "FMT_Utils/DFT_FMT_3D.py"]
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {Path(p).name: sha(p) for p in sources}, "config_sha256": sha(config),
            "hostname": socket.gethostname(), "python": platform.python_version()}


def inspect_sources(spec, config):
    from vtk.util.numpy_support import vtk_to_numpy
    report = {"version": spec["version"], "identity": identity(config), "sources": {}}
    for role in ("flow", "gt"):
        path = Path(spec[role])
        grid = read_vtk(path)
        row = {"path": str(path), "sha256": sha(path), "bounds": list(grid.GetBounds()),
               "points": grid.GetNumberOfPoints(), "cells": grid.GetNumberOfCells(), "arrays": {}}
        for association, attributes in (("point", grid.GetPointData()), ("cell", grid.GetCellData())):
            row["arrays"][association] = []
            for i in range(attributes.GetNumberOfArrays()):
                array = attributes.GetArray(i)
                if array is None:
                    continue
                entry = {"name": array.GetName(), "components": array.GetNumberOfComponents()}
                if array.GetName() == "VortexIds":
                    values, counts = np.unique(vtk_to_numpy(array), return_counts=True)
                    entry.update(values=values.tolist(), counts=counts.tolist())
                row["arrays"][association].append(entry)
        report["sources"][role] = row
    report["taxonomy_labels_available"] = False
    report["limitation"] = "VortexIds are instance identifiers, not the four human quality classes."
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "source_inspection.json", report)
    print(json.dumps({"inspection": str(out / "source_inspection.json"), "taxonomy_labels_available": False}))


def spatial_groups(boxes, margin):
    """Conservative connected overlap groups, used BEFORE labels or model fitting."""
    boxes = np.asarray(boxes, dtype=float)
    parents = list(range(len(boxes)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i in range(len(boxes)):
        for j in range(i):
            if np.all(boxes[i, 0] <= boxes[j, 1] + margin) and np.all(boxes[j, 0] <= boxes[i, 1] + margin):
                parents[root(i)] = root(j)
    return [f"overlap_{root(i):05d}" for i in range(len(boxes))]


def build_bundles(spec, config):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from scipy import ndimage
    out = Path(spec["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "bundles.npz").exists() or (out / "annotations.csv").exists():
        raise FileExistsError("Prepared bundles or annotations exist; do not overwrite them")
    grid = read_vtk(spec["flow"])
    if not grid.IsA("vtkStructuredGrid"):
        raise ValueError("Expected the original structured Channel flow")
    dims = [0, 0, 0]
    grid.GetDimensions(dims)
    nx, ny, nz = dims
    coordinates = vtk_to_numpy(grid.GetPoints().GetData()).reshape(nz, ny, nx, 3)
    axes = [coordinates[0, 0, :, 0], coordinates[0, :, 0, 1], coordinates[:, 0, 0, 2]]
    if any(np.any(np.diff(axis) <= 0) for axis in axes):
        raise ValueError("Channel axes must increase in x, y, z order")
    arrays = grid.GetPointData()
    for name in ("lambda2", "oyf", "vorticity"):
        if arrays.GetArray(name) is None:
            raise ValueError(f"Missing source field {name}")
    lambda2 = vtk_to_numpy(arrays.GetArray("lambda2")).reshape(nz, ny, nx)
    oyf = vtk_to_numpy(arrays.GetArray("oyf")).reshape(nz, ny, nx)
    options = spec["bundles"]
    candidate_mask = (lambda2 < options["lambda2_threshold"]) & (oyf > 0)
    components, component_count = ndimage.label(candidate_mask)
    sizes = np.bincount(components.ravel())
    eligible = np.flatnonzero(sizes >= options["minimum_head_points"])
    eligible = eligible[eligible != 0]
    # Coverage sampling is frozen before inspecting class labels or predictions.
    rng = np.random.default_rng(spec["preparation_seed"])
    chosen = eligible
    if len(chosen) > options["maximum_candidates"]:
        chosen = np.sort(rng.choice(chosen, options["maximum_candidates"], replace=False))
    regions = ndimage.find_objects(components)
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(grid)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "lambda2")
    threshold.SetLowerThreshold(options["lambda2_threshold"])
    threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_LOWER)
    threshold.AllScalarsOn()
    threshold.Update()
    vortex_grid = threshold.GetOutput()
    if not vortex_grid.GetNumberOfCells():
        raise ValueError("No cells survive the frozen lambda2 threshold")
    wall_span = float(axes[2][-1] - axes[2][0])
    all_x, all_masks, all_seeds, rows, rejected = [], [], [], [], []
    for component in chosen:
        box = regions[int(component) - 1]
        local_indices = np.argwhere(components[box] == component)
        indices = local_indices + np.array([s.start for s in box])
        if len(indices) > options["maximum_seed_points"]:
            take = np.sort(rng.choice(len(indices), options["maximum_seed_points"], replace=False))
            indices = indices[take]
        seeds = coordinates[tuple(indices.T)]
        lines, starts = trace_vortex_lines(vortex_grid, seeds,
            maximum_length=options["maximum_length_wall_span"] * wall_span,
            initial_step=options["initial_step_wall_span"] * wall_span,
            maximum_error=options["maximum_error_wall_span"] * wall_span)
        try:
            x, mask, seed_coordinates, normalization = normalize_bundle(lines, starts,
                max_lines=256, points=32, seed=spec["preparation_seed"] + int(component))
        except ValueError as error:
            rejected.append({"head_component": int(component), "reason": str(error)})
            continue
        dense = np.concatenate(lines)
        record = {"bundle_id": f"channel_head_{int(component):06d}", "head_component": int(component),
                  "head_point_count": int(sizes[component]), "traced_line_count": len(lines),
                  "bounds": [dense.min(0).tolist(), dense.max(0).tolist()], **normalization}
        all_x.append(x)
        all_masks.append(mask)
        all_seeds.append(seed_coordinates)
        rows.append(record)
        print(f"Prepared {record['bundle_id']}: {int(mask.sum())} lines", flush=True)
    if not rows:
        write_json(out / "preparation_failure.json", {"eligible_heads": len(eligible), "rejected": rejected})
        raise ValueError("No valid bundles; inspect preparation_failure.json without retuning on labels")
    margin = np.array([np.diff(axis).max() for axis in axes]) * 2
    groups = spatial_groups([row["bounds"] for row in rows], margin)
    for row, group in zip(rows, groups):
        row["group_id"] = group
    np.savez_compressed(out / "bundles.npz", x=np.stack(all_x), mask=np.stack(all_masks),
                        seeds=np.stack(all_seeds), bundle_ids=np.array([r["bundle_id"] for r in rows]))
    with (out / "annotations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bundle_id", "group_id", "label", "split", "annotator"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"bundle_id": row["bundle_id"], "group_id": row["group_id"]})
    report = {"version": spec["version"], "identity": identity(config), "flow_sha256": sha(spec["flow"]),
              "bundle_file_sha256": sha(out / "bundles.npz"), "head_components": component_count,
              "eligible_heads": len(eligible), "selected_heads": chosen.tolist(), "rejected": rejected,
              "overlap_margin_xyz": margin.tolist(), "group_count": len(set(groups)), "bundles": rows,
              "labels": "unlabeled; GT was not used for candidate selection or line tracing",
              "candidate_method": "lambda2 plus positive stored oyf, connected heads; local adaptation, not full Zafar segmentation"}
    write_json(out / "bundle_manifest.json", report)
    print(json.dumps({"bundles": len(rows), "overlap_groups": len(set(groups)), "labels": "pending"}))


def load_labeled_data(out):
    manifest = json.loads((out / "bundle_manifest.json").read_text(encoding="utf-8"))
    if sha(out / "bundles.npz") != manifest["bundle_file_sha256"]:
        raise ValueError("Frozen bundle data changed")
    with (out / "annotations.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {row["bundle_id"]: row["group_id"] for row in manifest["bundles"]}
    if len(rows) != len(expected) or len({r["bundle_id"] for r in rows}) != len(rows):
        raise ValueError("Annotations must contain every unique candidate exactly once")
    by_id = {r["bundle_id"]: r for r in rows}
    group_splits = {}
    for row in rows:
        if row["bundle_id"] not in expected or row["group_id"] != expected[row["bundle_id"]]:
            raise ValueError("Bundle or conservative overlap-group identity changed")
        if row["label"] not in CLASS_NAMES or row["split"] not in {"train", "validation", "test"} or not row["annotator"].strip():
            raise ValueError("Four-class human annotations, annotator and train/validation/test split are required; no GT-to-class inference")
        group_splits.setdefault(row["group_id"], set()).add(row["split"])
    if any(len(value) != 1 for value in group_splits.values()):
        raise ValueError("Overlapping bundle/source groups cross dataset splits")
    data = np.load(out / "bundles.npz", allow_pickle=False)
    ordered = [by_id[str(i)] for i in data["bundle_ids"]]
    splits = np.array([r["split"] for r in ordered])
    labels = np.array([CLASS_NAMES.index(r["label"]) for r in ordered])
    for split in ("train", "validation", "test"):
        if set(labels[splits == split]) != set(range(4)):
            raise ValueError(f"{split} lacks one or more taxonomy classes; cannot report four-class performance")
    return {k: data[k] for k in data.files}, labels, splits, manifest


def export_review(spec, config):
    """Export resampled curves in physical coordinates for ParaView review."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    out = Path(spec["output"])
    manifest = json.loads((out / "bundle_manifest.json").read_text(encoding="utf-8"))
    if sha(out / "bundles.npz") != manifest["bundle_file_sha256"]:
        raise ValueError("Frozen bundle data changed")
    with np.load(out / "bundles.npz", allow_pickle=False) as data:
        x, mask, identifiers = data["x"], data["mask"], data["bundle_ids"]
    records = {r["bundle_id"]: r for r in manifest["bundles"]}
    geometry = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    coordinates, bundle_ids, line_ids, names = [], [], [], []
    offset = 0
    for b, bundle_id in enumerate(identifiers):
        record = records[str(bundle_id)]
        for line_id in np.flatnonzero(mask[b]):
            line = x[b, line_id, :, :3].astype(np.float64) * record["radius"] + record["centroid"]
            coordinates.append(line)
            cells.InsertNextCell(len(line))
            for j in range(len(line)):
                cells.InsertCellPoint(offset + j)
            offset += len(line)
            bundle_ids.append(b)
            line_ids.append(line_id)
            names.append(str(bundle_id))
    points.SetData(numpy_to_vtk(np.concatenate(coordinates), deep=True))
    geometry.SetPoints(points)
    geometry.SetLines(cells)
    for name, values in (("bundle_index", bundle_ids), ("line_index", line_ids)):
        array = numpy_to_vtk(np.asarray(values, dtype=np.int32), deep=True)
        array.SetName(name)
        geometry.GetCellData().AddArray(array)
    array = vtk.vtkStringArray()
    array.SetName("bundle_id")
    for name in names:
        array.InsertNextValue(name)
    geometry.GetCellData().AddArray(array)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out / "bundle_review.vtp"))
    writer.SetInputData(geometry)
    if writer.Write() != 1:
        raise RuntimeError("Failed to export candidate review geometry")
    print(json.dumps({"review": str(out / "bundle_review.vtp"), "bundles": len(identifiers),
                      "lines": len(bundle_ids), "coordinates": "original physical x/y/z", "labels": "not inferred"}))


def augment(x, head_seeds, y, generator, reverse_probability=0.25):
    """Geometry-only augmentation shared across all arms; applied before FMT/PCA."""
    result = x.copy()
    seeds = head_seeds.copy()
    labels = y.copy()
    for i, label in enumerate(y):
        matrix = np.eye(3)
        if label == 0:
            if generator.random() < reverse_probability:
                # Right-hand +90-degree rotation about +Y after X reflection.
                matrix = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]) @ np.diag([-1., 1., 1.])
                labels[i] = 2
            else:
                if generator.random() < 0.5:
                    matrix[1, 1] = -1
                angle = generator.uniform(-15, 15) * np.pi / 180
                rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
                matrix = rotation @ matrix
        elif label == 3:
            for axis in range(3):
                angle = generator.uniform(-np.pi, np.pi)
                others = [a for a in range(3) if a != axis]
                rotation = np.eye(3)
                rotation[np.ix_(others, others)] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                matrix = rotation @ matrix
            matrix = np.diag(generator.choice([-1., 1.], 3)) @ matrix
        result[i, :, :, :3] = result[i, :, :, :3] @ matrix.T
        result[i, :, :, 3:] = result[i, :, :, 3:] @ matrix.T
        seeds[i] = seeds[i] @ matrix.T
    return result, seeds, labels


def train_comparison(spec, config):
    # Validate the complete supervision and split BEFORE any model initialization.
    out = Path(spec["output"])
    data, labels, splits, manifest = load_labeled_data(out)
    import torch
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score
    from torch.nn import functional as F
    options = spec["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(options["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    train_ids, validation_ids, test_ids = [np.flatnonzero(splits == s) for s in ("train", "validation", "test")]
    x, mask, seeds = data["x"], data["mask"], data["seeds"]
    weights = inverse_frequency_weights(torch.from_numpy(labels[train_ids])).to(device)
    batch_size = options["batch_size"]
    results = []
    for run_seed in options["seeds"]:
        for method in spec["methods"]:
            destination = out / "runs" / f"{method}_seed{run_seed}"
            if destination.exists():
                raise FileExistsError(f"Completed/partial run exists: {destination}")
            destination.mkdir(parents=True)
            start = time.time()
            started_at = datetime.now(timezone.utc).isoformat()
            torch.manual_seed(run_seed)
            rng = np.random.default_rng(run_seed)
            model = PaperAutoencoder().to(device)
            adapter = None if method == "paper_bilstm" else AuxiliaryEncoder(model.encoder).to(device)
            pca = scaler = None
            if adapter is not None:
                if method == "raw_pca_residual":
                    raw_training = x[train_ids][mask[train_ids]].reshape(-1, 192)
                    if len(raw_training) < 161:
                        raise ValueError("Too few training lines for the frozen PCA161 control")
                    pca = PCA(161, svd_solver="full").fit(raw_training)
                    training_features = pca.transform(raw_training)
                elif method == "fmt_residual":
                    training_features = fmt_bundle_features(x[train_ids], mask[train_ids], seeds[train_ids]).numpy()[mask[train_ids]]
                else:
                    raise ValueError(method)
                scaler = StandardScaler().fit(training_features)

            def encode(batch_x, batch_mask, batch_seeds):
                tx = torch.as_tensor(batch_x, device=device)
                if adapter is None:
                    return model.encoder(tx)
                auxiliary = np.zeros((*batch_mask.shape, 161), np.float32)
                if pca is not None:
                    values = pca.transform(batch_x[batch_mask].reshape(-1, 192))
                else:
                    values = fmt_bundle_features(batch_x, batch_mask, batch_seeds).numpy()[batch_mask]
                auxiliary[batch_mask] = scaler.transform(values).astype(np.float32)
                return adapter(tx, torch.as_tensor(auxiliary, device=device), torch.as_tensor(batch_mask, device=device))

            # Adapter shares the base encoder; avoid duplicating optimizer parameters.
            encoder = model.encoder if adapter is None else adapter
            pretrain_parameters = list(encoder.parameters()) + list(model.decoder.parameters())
            optimizer = torch.optim.AdamW(pretrain_parameters, lr=options["pretrain_learning_rate"], weight_decay=options["weight_decay"])
            best_reconstruction, best_pretrained, history = float("inf"), None, []
            for epoch in range(1, options["pretrain_epochs"] + 1):
                model.train()
                if adapter is not None:
                    adapter.train()
                order = rng.permutation(train_ids)
                losses = []
                for offset in range(0, len(order), batch_size):
                    ids = order[offset:offset + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    z = encode(x[ids], mask[ids], seeds[ids])
                    prediction = model.decoder(z, x.shape[1], x.shape[2])
                    loss = reconstruction_loss(prediction, torch.as_tensor(x[ids], device=device), torch.as_tensor(mask[ids], device=device))
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite pretraining loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pretrain_parameters, options["gradient_clip"])
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                if adapter is not None:
                    adapter.eval()
                total = 0.
                with torch.no_grad():
                    for offset in range(0, len(validation_ids), batch_size):
                        ids = validation_ids[offset:offset + batch_size]
                        prediction = model.decoder(encode(x[ids], mask[ids], seeds[ids]), x.shape[1], x.shape[2])
                        total += len(ids) * float(reconstruction_loss(prediction, torch.as_tensor(x[ids], device=device), torch.as_tensor(mask[ids], device=device)))
                score = total / len(validation_ids)
                history.append({"stage": "pretrain", "epoch": epoch, "training_loss": float(np.mean(losses)), "validation_loss": score})
                if score < best_reconstruction:
                    best_reconstruction, best_pretrained = score, copy.deepcopy(encoder.state_dict())
            encoder.load_state_dict(best_pretrained)
            # Extra adapter construction must not change classifier initialization.
            torch.manual_seed(run_seed + 100000)
            classifier = AsymmetricMarginClassifier().to(device)
            parameters = list(encoder.parameters()) + list(classifier.parameters())
            optimizer = torch.optim.AdamW(parameters, lr=options["fine_tune_learning_rate"], weight_decay=options["weight_decay"])
            best_f1, best_epoch, best_state = -1., 0, None

            def predict(ids):
                encoder.eval()
                classifier.eval()
                probabilities = []
                with torch.no_grad():
                    for offset in range(0, len(ids), batch_size):
                        chosen = ids[offset:offset + batch_size]
                        z = encode(x[chosen], mask[chosen], seeds[chosen])
                        probabilities.append(classifier(z).softmax(-1).cpu().numpy())
                return np.concatenate(probabilities)

            for epoch in range(1, options["fine_tune_epochs"] + 1):
                encoder.train()
                classifier.train()
                order = rng.permutation(train_ids)
                losses = []
                for offset in range(0, len(order), batch_size):
                    ids = order[offset:offset + batch_size]
                    ax, a_seeds, ay = augment(x[ids], seeds[ids], labels[ids], rng, options["reverse_probability"])
                    optimizer.zero_grad(set_to_none=True)
                    ty = torch.as_tensor(ay, dtype=torch.long, device=device)
                    logits = classifier(encode(ax, mask[ids], a_seeds), ty)
                    loss = F.cross_entropy(logits, ty, weight=weights)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite classification loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, options["gradient_clip"])
                    optimizer.step()
                    losses.append(float(loss.detach()))
                probabilities = predict(validation_ids)
                score = float(f1_score(labels[validation_ids], probabilities.argmax(1), average="macro", labels=range(4), zero_division=0))
                history.append({"stage": "finetune", "epoch": epoch, "training_loss": float(np.mean(losses)), "validation_macro_f1": score})
                if score > best_f1:
                    best_f1, best_epoch = score, epoch
                    best_state = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(classifier.state_dict()))
                if epoch >= options["warmup_epochs"] and epoch - best_epoch >= options["patience"]:
                    break
            encoder.load_state_dict(best_state[0])
            classifier.load_state_dict(best_state[1])
            # Test data enters inference only after validation has selected the model.
            probabilities = predict(test_ids)
            predicted = probabilities.argmax(1)
            record = {"method": method, "seed": run_seed, "parameter_count": sum(p.numel() for p in parameters),
                      "best_validation_macro_f1": best_f1, "selected_epoch": best_epoch,
                      "test_macro_f1": float(f1_score(labels[test_ids], predicted, average="macro", labels=range(4), zero_division=0)),
                      "test_per_class_f1": f1_score(labels[test_ids], predicted, average=None, labels=range(4), zero_division=0).tolist(),
                      "test_balanced_accuracy": float(balanced_accuracy_score(labels[test_ids], predicted)),
                      "test_confusion_matrix": confusion_matrix(labels[test_ids], predicted, labels=range(4)).tolist(),
                      "training_bundles": len(train_ids), "validation_bundles": len(validation_ids), "test_bundles": len(test_ids),
                      "class_names": list(CLASS_NAMES),
                      "class_counts": {name: np.bincount(labels[ids], minlength=4).tolist() for name, ids in
                                       (("train", train_ids), ("validation", validation_ids), ("test", test_ids))},
                      "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                      "device": str(device), "torch": torch.__version__, "seconds": time.time() - start,
                      "annotation_sha256": sha(out / "annotations.csv"), "identity": identity(config), "history": history,
                      "complete_config": spec,
                      "bundle_file_sha256": manifest["bundle_file_sha256"], "checkpoint_files_written": 0}
            np.savez_compressed(destination / "predictions.npz", probabilities=probabilities, labels=labels[test_ids],
                                bundle_ids=data["bundle_ids"][test_ids],
                                confidence=structural_confidence(torch.from_numpy(probabilities)).numpy())
            write_json(destination / "result.json", record)
            results.append({k: record[k] for k in ("method", "seed", "parameter_count", "test_macro_f1", "test_balanced_accuracy")})
            print(json.dumps(results[-1]), flush=True)
    summary = {}
    for method in spec["methods"]:
        values = np.array([r["test_macro_f1"] for r in results if r["method"] == method])
        summary[method] = {"seeds": len(values), "test_macro_f1_mean": float(values.mean()),
                           "test_macro_f1_sample_std": float(values.std(ddof=1)) if len(values) > 1 else None}
    write_json(out / "results.json", {"version": spec["version"], "runs": results, "summary": summary,
               "evaluation": "single-snapshot grouped bundle classification, not the paper's surface-IoU benchmark"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("inspect", "build", "review", "train"))
    parser.add_argument("--config", default="config/mainExp_Task4C_BundleQuality_1.1.json")
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text(encoding="utf-8"))
    {"inspect": inspect_sources, "build": build_bundles, "review": export_review, "train": train_comparison}[args.phase](spec, args.config)


if __name__ == "__main__":
    main()
