import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Select_Task3_F22AnchoredFeatures import _fast_screen_shortlist


def _write_fast_screen(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("candidate_id", "f1_gain", "average_precision_gain"),
        )
        writer.writeheader()
        writer.writerows(rows)


class FastScreenShortlistTest(unittest.TestCase):
    def test_ranks_by_weaker_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fast_screen.csv"
            _write_fast_screen(path, [
                {
                    "candidate_id": "a",
                    "f1_gain": 0.20,
                    "average_precision_gain": 0.01,
                },
                {
                    "candidate_id": "b",
                    "f1_gain": 0.08,
                    "average_precision_gain": 0.07,
                },
                {
                    "candidate_id": "c",
                    "f1_gain": 0.06,
                    "average_precision_gain": 0.09,
                },
                {
                    "candidate_id": "d",
                    "f1_gain": 0.10,
                    "average_precision_gain": 0.04,
                },
            ])

            self.assertEqual(
                _fast_screen_shortlist(path, top_k=2), ["b", "c"]
            )

    def test_rejects_invalid_top_k(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fast_screen.csv"
            _write_fast_screen(path, [{
                "candidate_id": "a",
                "f1_gain": 0.1,
                "average_precision_gain": 0.1,
            }])

            with self.assertRaisesRegex(ValueError, "must be positive"):
                _fast_screen_shortlist(path, top_k=0)
            with self.assertRaisesRegex(RuntimeError, "fewer than top_k"):
                _fast_screen_shortlist(path, top_k=2)


if __name__ == "__main__":
    unittest.main()
