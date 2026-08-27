"""Generate Task3-4.1 configs that differ only in the IVD percentile label."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from Build_Task23_IVDPercentile_Labels import percentile_tag


POPULATIONS = ("old8", "new2")


def _load(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _write_immutable(path: Path, value: dict, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = yaml.safe_load(path.read_text(encoding="utf-8"))
        if previous != value and not overwrite:
            raise RuntimeError(f"generated config changed: {path}")
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _label_root(master: dict, split: str, population: str, tag: str) -> str:
    return str(master["label_roots"][split][population]).format(tag=tag)


def prepare(config_path: str, overwrite: bool = False) -> Path:
    master = _load(config_path)
    task3 = master["task3"]
    base_main = _load(task3["main_config"])
    base_search = _load(task3["search_config"])
    generated = Path(task3["generated_config_dir"])
    output_root = Path(task3["output_root"])
    manifest = {
        "experiment": master["experiment"],
        "frozen_stage2_selection": base_main["stage2_selection"],
        "frozen_recipe_manifest": base_main["frozen_recipe_manifest"],
        "percentiles": [],
        "configs": {},
    }
    for value_ in master["requested_percentiles"]:
        value = float(value_)
        tag = percentile_tag(value)
        percentile_root = output_root / tag
        manifest["percentiles"].append({"value": value, "tag": tag})
        manifest["configs"][tag] = {}

        for population in POPULATIONS:
            baseline = _load(task3["baseline_templates"][population])
            baseline["experiment"] = (
                f"{master['experiment']}_Task3_{tag}_baselines_{population}"
            )
            baseline["label_cache_root"] = _label_root(
                master, "development", population, tag
            )
            baseline["expected_ivd_percentile"] = value
            baseline["output_dir"] = (
                percentile_root / f"development_{population}" / "baselines"
            ).as_posix()
            path = generated / f"task3_{tag}_baselines_{population}.yaml"
            _write_immutable(path, baseline, overwrite)
            manifest["configs"][tag][f"baseline_{population}"] = path.as_posix()

        search = json.loads(json.dumps(base_search))
        search["experiment"] = f"{master['experiment']}_Task3_{tag}_frozen_search"
        search["output_root"] = (
            percentile_root / "development_frozen_search"
        ).as_posix()
        search["expected_ivd_percentile"] = value
        search["require_source_reference_match"] = False
        for group in search["groups"].values():
            population = (
                "new2" if set(group["datasets"]) <= {"boeing747", "smokeBuoyancy"}
                else "old8"
            )
            group["label_cache_root"] = _label_root(
                master, "development", population, tag
            )
            group["raw_checkpoint_dir"] = (
                percentile_root / f"development_{population}"
                / "baselines" / "checkpoints"
            ).as_posix()
        search_path = generated / f"task3_{tag}_search.yaml"
        _write_immutable(search_path, search, overwrite)
        manifest["configs"][tag]["search"] = search_path.as_posix()

        final = json.loads(json.dumps(base_main))
        final["experiment"] = f"{master['experiment']}_Task3_{tag}_confirmation"
        final["search_config"] = search_path.as_posix()
        final["output_root"] = (percentile_root / "final_confirmation").as_posix()
        final["expected_ivd_percentile"] = value
        final["require_confirmation_reference_match"] = False
        for population, group in final["confirmation_roots"].items():
            group["label_root"] = _label_root(
                master, "confirmation", population, tag
            )
        final_path = generated / f"task3_{tag}_final.yaml"
        _write_immutable(final_path, final, overwrite)
        manifest["configs"][tag]["final"] = final_path.as_posix()

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
        "--config", default="config/Ablation_Task23IVDPercentile_1.2.yaml"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(args.config, args.overwrite)
