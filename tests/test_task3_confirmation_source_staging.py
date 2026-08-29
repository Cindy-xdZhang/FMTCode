import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import netCDF4 as nc
import numpy as np

import Build_Task3_SpatialRobust_Confirmation_5_2 as spatial
from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d
from Prepare_Task3_Confirmation_SourcePacks_12_2 import (
    PACK_DATASETS,
    _remote_posix_path,
    _write_pack,
    rewrite_manifest_remote_paths,
)


def _all_dataset_indices():
    return {
        dataset: [int(value) for value in indices]
        for settings in spatial.SETTINGS.values()
        for dataset, indices in settings["indices"].items()
    }


class Task3ConfirmationSourceStagingTests(unittest.TestCase):
    def _manifest(self, root: Path):
        entries = {}
        for dataset, original in _all_dataset_indices().items():
            source = root / f"{dataset}.nc"
            source.write_bytes(dataset.encode("utf-8"))
            entries[dataset] = {
                "kind": "test_source",
                "path": str(source),
                "original_fixed_indices": original,
                "effective_fixed_indices": [0, 14, 28, 42],
            }
        manifest = root / "source_staging_manifest.json"
        manifest.write_text(json.dumps({
            "experiment": "test_source_staging",
            "scientific_protocol_unchanged": True,
            "seed_grid_phase": list(spatial.SEED_GRID_PHASE),
            "datasets": entries,
        }), encoding="utf-8")
        return manifest

    def test_cache_config_remaps_only_operational_paths_and_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            with patch.dict(
                "os.environ",
                {spatial.SOURCE_STAGING_ENV: str(manifest)},
            ):
                config = spatial._cache_config("old8")
            expected_original = spatial.SETTINGS["old8"]["indices"]
            self.assertEqual(
                dict(config.sampling.original_fixed_time_indices_by_dataset),
                expected_original,
            )
            self.assertTrue(all(
                list(values) == [0, 14, 28, 42]
                for values in
                config.sampling.fixed_time_indices_by_dataset.values()
            ))
            for item in config.datasets:
                self.assertEqual(
                    Path(item["path"]).name, f"{item['id']}.nc"
                )

    def test_cache_config_rejects_changed_original_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["datasets"]["cylinder3d"]["original_fixed_indices"][0] += 1
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with patch.dict(
                "os.environ",
                {spatial.SOURCE_STAGING_ENV: str(manifest)},
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "changed original time indices"
                ):
                    spatial._cache_config("old8")

    def test_pack_is_exact_and_refuses_implicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.nc"
            output = root / "packed.nc"
            with nc.Dataset(source, "w", format="NETCDF4") as dataset:
                for name, count in (
                    ("tdim", 10), ("zdim", 2),
                    ("ydim", 3), ("xdim", 4),
                ):
                    dataset.createDimension(name, count)
                    coordinate = dataset.createVariable(name, "f8", (name,))
                    if name == "tdim":
                        coordinate[:] = np.arange(count) * 0.1
                    else:
                        coordinate[:] = np.linspace(-1.0, 1.0, count)
                base = np.arange(10 * 2 * 3 * 4, dtype=np.float32).reshape(
                    10, 2, 3, 4
                )
                for name, values in (
                    ("u", base), ("v", base + 1000.0), ("w", -base),
                ):
                    variable = dataset.createVariable(
                        name, "f4", ("tdim", "zdim", "ydim", "xdim")
                    )
                    variable[:] = values

            details = _write_pack(
                "test", source, output, [1, 5], 3, 96
            )
            self.assertEqual(details["effective_fixed_indices"], [0, 3])
            self.assertTrue(details["all_windows_verified_exact"])
            for original, effective in ((1, 0), (5, 3)):
                expected, _ = load_netcdf_window_3d(
                    source, original, 3, 96
                )
                observed, _ = load_netcdf_window_3d(
                    output, effective, 3, 96
                )
                np.testing.assert_array_equal(
                    observed.field, expected.field
                )
            with self.assertRaises(FileExistsError):
                _write_pack("test", source, output, [1, 5], 3, 96)

    def test_remote_paths_are_posix_even_when_generated_on_windows(self):
        self.assertEqual(
            str(_remote_posix_path(
                r"\ibex\scratch\zhanx0o\task3-confirmation"
            )),
            "/ibex/scratch/zhanx0o/task3-confirmation",
        )
        self.assertEqual(
            str(_remote_posix_path(
                r"\ibex\scratch\zhanx0o\task3-confirmation\channel.vtk"
            )),
            "/ibex/scratch/zhanx0o/task3-confirmation/channel.vtk",
        )
        with self.assertRaisesRegex(ValueError, "absolute POSIX"):
            _remote_posix_path("relative/source.nc")

    def test_manifest_path_rewrite_does_not_change_scientific_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            entries = {}
            for dataset, indices in _all_dataset_indices().items():
                if dataset in PACK_DATASETS:
                    kind = "exact_prestrided_temporal_window_pack"
                elif dataset == "channel":
                    kind = "original_channel_vtk"
                else:
                    kind = "remote_original_netcdf"
                entries[dataset] = {
                    "kind": kind,
                    "path": rf"\wrong\windows\{dataset}",
                    "original_fixed_indices": indices,
                    "effective_fixed_indices": list(indices),
                    "sentinel": f"unchanged-{dataset}",
                }
            original = {
                "experiment": "test",
                "scientific_protocol_unchanged": True,
                "seed_grid_phase": list(spatial.SEED_GRID_PHASE),
                "datasets": entries,
                "equivalence": "unchanged",
            }
            manifest.write_text(
                json.dumps(original, sort_keys=True), encoding="utf-8"
            )
            rewrite_manifest_remote_paths(
                manifest,
                r"\ibex\scratch\zhanx0o\packs",
                r"\home\zhanx0o\data",
                r"\ibex\scratch\zhanx0o\packs\channel.vtk",
            )
            repaired = json.loads(manifest.read_text(encoding="utf-8"))
            for dataset, entry in repaired["datasets"].items():
                self.assertTrue(entry["path"].startswith("/"), dataset)
                self.assertNotIn("\\", entry["path"], dataset)
                expected = dict(original["datasets"][dataset])
                expected.pop("path")
                observed = dict(entry)
                observed.pop("path")
                self.assertEqual(observed, expected)
            for key in ("experiment", "scientific_protocol_unchanged",
                        "seed_grid_phase", "equivalence"):
                self.assertEqual(repaired[key], original[key])


if __name__ == "__main__":
    unittest.main()
