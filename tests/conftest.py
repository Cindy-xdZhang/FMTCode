"""Test compatibility for experiment scripts moved out of the repository root."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPOSITORY_ROOT / "experiments"

for path in (REPOSITORY_ROOT, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
