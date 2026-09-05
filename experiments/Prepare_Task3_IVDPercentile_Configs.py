"""Generate Task3 configs that differ only in the whole-field IVD percentile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from Build_Task23_IVDPercentile_Labels import percentile_tag


GROUPS = ("old8", "new2")
MODES = ("fmt", "raw_pca")


def _load(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _write_immutable(path: Path, value: dict, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = yaml.safe_load(path.read_text(encoding="utf-8"))
        if previous != value and not overwrite:
            raise RuntimeError(f"generated config changed: {path}")
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _label_root(master: dict, tag: str, phase: str, group: str) -> str:
    return (
        Path(master["output_dir"]) / "labels" / tag
        / f"{phase}_{group}" / "labels"
    ).as_posix()


def _experiment(master: dict, tag: str, component: str, group: str = "") -> str:
    suffix = f"_{group}" if group else ""
    return f"{master['experiment']}_Task3_{tag}_{component}{suffix}"


def prepare(config_path: str, overwrite: bool = False) -> Path:
    master = _load(config_path)
    task3 = master["task3"]
    generated = Path(task3["generated_config_dir"])
    root = Path(task3["output_root"])
    manifest = {
        "experiment": master["experiment"],
        "percentiles": [],
        "configs": {},
    }
    for percentile_value in master["requested_percentiles"]:
        percentile = float(percentile_value)
        tag = percentile_tag(percentile)
        pct_root = root / tag
        manifest["percentiles"].append({"value": percentile, "tag": tag})
        manifest["configs"][tag] = {}
        for group in GROUPS:
            baseline = _load(task3["templates"][f"baseline_{group}"])
            baseline["experiment"] = _experiment(
                master, tag, "baselines", group
            )
            baseline["label_cache_root"] = _label_root(
                master, tag, "development", group
            )
            baseline["output_dir"] = (
                pct_root / f"development_{group}" / "baselines"
            ).as_posix()
            baseline_path = generated / f"task3_{tag}_baselines_{group}.yaml"
            _write_immutable(baseline_path, baseline, overwrite)
            manifest["configs"][tag][f"baseline_{group}"] = baseline_path.as_posix()

            for mode in MODES:
                residual = _load(task3["templates"][f"{mode}_{group}"])
                residual["experiment"] = _experiment(
                    master, tag, f"{mode}_residual", group
                )
                residual["label_cache_root"] = _label_root(
                    master, tag, "development", group
                )
                residual["raw_checkpoint_dir"] = (
                    pct_root / f"development_{group}" / "baselines" / "checkpoints"
                ).as_posix()
                residual["output_dir"] = (
                    pct_root / f"development_{group}" / f"{mode}_residual"
                ).as_posix()
                residual_path = generated / f"task3_{tag}_{mode}_{group}.yaml"
                _write_immutable(residual_path, residual, overwrite)
                manifest["configs"][tag][f"{mode}_{group}"] = residual_path.as_posix()

        evaluate = _load(task3["templates"]["evaluate"])
        evaluate["experiment"] = _experiment(master, tag, "evaluate")
        evaluate["output_dir"] = (pct_root / "final_confirmation").as_posix()
        for group_spec in evaluate["groups"]:
            group = str(group_spec["name"])
            group_spec["label_cache_root"] = _label_root(
                master, tag, "confirmation", group
            )
            development_root = pct_root / f"development_{group}"
            group_spec["baseline_checkpoint_roots"] = [
                (development_root / "baselines" / "checkpoints").as_posix()
            ]
            group_spec["raw_pca_checkpoint_roots"] = [
                (development_root / "raw_pca_residual" / "checkpoints").as_posix()
            ]
            group_spec["fmt_checkpoint_roots"] = [
                (development_root / "fmt_residual" / "checkpoints").as_posix()
            ]
            group_spec["development_result_csvs"] = [
                (development_root / "baselines" / "per_run.csv").as_posix(),
                (development_root / "raw_pca_residual" / "per_run.csv").as_posix(),
            ]
        evaluate_path = generated / f"task3_{tag}_evaluate.yaml"
        _write_immutable(evaluate_path, evaluate, overwrite)
        manifest["configs"][tag]["evaluate"] = evaluate_path.as_posix()

    manifest_path = generated / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest and not overwrite:
            raise RuntimeError(f"generated manifest changed: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/Ablation_Task23IVDPercentile_1.1.yaml"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(args.config, args.overwrite)
