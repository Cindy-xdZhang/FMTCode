"""Independently audit the sealed Task3 8.1 confirmation artifacts.

This wrapper reuses only the implementation-independent 7.2 evidence auditor;
it does not import the 8.1 confirmation, evaluation, or summary code.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import Audit_Task3_AdaptiveTuned_7_2 as _base


EXPECTED_EXPERIMENT = "mainExp_Task3_3D_8.1"
EXPECTED_SOURCE_EXPERIMENT = "Verify_Task3_ExtendedPortfolio_54.1"


@contextmanager
def _configured_base():
    previous_experiment = _base.EXPECTED_EXPERIMENT
    previous_source = _base.EXPECTED_SOURCE_EXPERIMENT
    try:
        _base.EXPECTED_EXPERIMENT = EXPECTED_EXPERIMENT
        _base.EXPECTED_SOURCE_EXPERIMENT = EXPECTED_SOURCE_EXPERIMENT
        yield
    finally:
        _base.EXPECTED_EXPERIMENT = previous_experiment
        _base.EXPECTED_SOURCE_EXPERIMENT = previous_source


def audit(config_path: str | Path, artifact_dir: str | Path,
          output: str | Path | None = None) -> dict:
    with _configured_base():
        return _base.audit(config_path, artifact_dir, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit(args.config, args.artifact_dir, args.output)


if __name__ == "__main__":
    main()
