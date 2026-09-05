"""Standard-library tests: python -m unittest discover -s tests -p test_research_archive.py."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "research_archive", Path(__file__).resolve().parents[1] / "tools/research_archive.py"
)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


class ResearchArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for directory in (*archive.DIRECTORIES, "docs", "FMT_Utils", "tools"):
            (self.root / directory).mkdir()
        for name, value in {
            "ROOT": self.root,
            "ARCHIVE": self.root / ".research_archive/history.zip",
            "MANIFEST": self.root / "docs/research_archive_manifest.json",
        }.items():
            patcher = patch.object(archive, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def source(self, name, text):
        (self.root / name).write_text(text, encoding="utf-8")

    def test_keep_transitive_dependencies_and_retire_obsolete_tests(self):
        self.source("experiments/Run_Main.py", "from experiments.Verify_Helper import fit")
        self.source("experiments/Verify_Helper.py", "CONFIG = 'config/Verify_Helper_1.1.yaml'")
        self.source("config/Verify_Helper_1.1.yaml", "seed: 1")
        self.source("config/Verify_Helper_1.2.yaml", "seed: 2")
        self.source("tests/test_old.py", "CONFIG = 'config/Verify_Helper_1.2.yaml'")
        self.source("tests/test_library.py", "import math")
        plan = archive.make_plan()
        self.assertEqual({e["path"] for e in plan["files"]}, {
            "config/Verify_Helper_1.2.yaml", "tests/test_old.py"
        })

    def test_archive_and_restore_exact_bytes(self):
        path = "config/Verify_Old_1.1.yaml"
        self.source(path, "seed: 1\n# historical bytes\n")
        original = (self.root / path).read_bytes()
        archive.apply_plan(archive.make_plan())
        self.assertFalse((self.root / path).exists())
        archive.verify_or_restore(False)
        archive.verify_or_restore(True)
        self.assertEqual((self.root / path).read_bytes(), original)
        archive.verify_or_restore(True)  # Idempotent, no overwrites.

    def test_archive_verification_and_restore_never_overwrite(self):
        path = "config/Verify_Old_1.1.yaml"
        self.source(path, "seed: 1")
        archive.apply_plan(archive.make_plan())
        self.source(path, "seed: 999")
        with self.assertRaises(FileExistsError):
            archive.verify_or_restore(True)
        self.assertEqual((self.root / path).read_text(), "seed: 999")
        with self.assertRaises(FileExistsError):
            archive.apply_plan(archive.make_plan())
        archive.ARCHIVE.write_bytes(b"corrupted")
        with self.assertRaises(ValueError):
            archive.verify_or_restore(False)

    def test_reject_source_change_before_removal(self):
        path = "config/Verify_Old_1.1.yaml"
        self.source(path, "seed: 1")
        plan = archive.make_plan()
        self.source(path, "seed: 2")
        with self.assertRaises(ValueError):
            archive.apply_plan(plan)
        self.assertTrue((self.root / path).exists())

    def test_reject_unsafe_paths_and_models(self):
        for name in ("../outside.py", "config/../../outside.py", "outputs/model.pt",
                     "config/model.pt", str(self.root / "config/absolute.yaml")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                archive.checked_path(name)


if __name__ == "__main__":
    unittest.main()
