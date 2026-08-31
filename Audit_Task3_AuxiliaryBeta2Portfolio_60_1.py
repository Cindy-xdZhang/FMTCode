"""Independently audit Task3 portfolio 60.1 without importing its selector."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import Audit_Task3_AuxiliaryLearningRatePortfolio_56_1 as _base


EXPERIMENT = "Verify_Task3_AuxiliaryBeta2Portfolio_60.1"
CURRENT_SOURCE_NAME = "current_portfolio"
AUXILIARY_SOURCE_NAME = "auxiliary_beta2"
AUXILIARY_ARCHIVE_COUNT = 660
CURRENT_SOURCE_LABEL = "58.1"
AUXILIARY_SOURCE_LABEL = "59.1"
PORTFOLIO_LABEL = "60.1"


@contextmanager
def _configured_contract():
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "CURRENT_SOURCE_NAME": CURRENT_SOURCE_NAME,
        "AUXILIARY_SOURCE_NAME": AUXILIARY_SOURCE_NAME,
        "AUXILIARY_ARCHIVE_COUNT": AUXILIARY_ARCHIVE_COUNT,
        "CURRENT_SOURCE_LABEL": CURRENT_SOURCE_LABEL,
        "AUXILIARY_SOURCE_LABEL": AUXILIARY_SOURCE_LABEL,
        "PORTFOLIO_LABEL": PORTFOLIO_LABEL,
    }
    previous = {name: getattr(_base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(_base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(_base, name, value)


def audit(config_path: Path, artifact_dir: Path, output_path: Path) -> dict:
    with _configured_contract():
        return _base.audit(config_path, artifact_dir, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output or arguments.artifact_dir / "independent_audit.json"
    result = audit(
        arguments.config.resolve(), arguments.artifact_dir.resolve(), output.resolve()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
