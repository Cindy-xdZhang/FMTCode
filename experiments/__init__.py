"""Versioned experiment entry points.

Historical scripts used flat imports such as ``import Verify_Task3_FMTResidual``.
Keeping this directory on ``sys.path`` preserves those imports while the scripts are
gradually converted to explicit package imports.
"""

from pathlib import Path
import sys


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
