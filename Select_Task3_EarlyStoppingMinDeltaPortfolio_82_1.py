"""Freeze the guarded Task3 winner between portfolios 80.1 and 81.1."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import Select_Task3_AuxiliaryLearningRatePortfolio_56_1 as _base


EXPERIMENT = "Verify_Task3_EarlyStoppingMinDeltaPortfolio_82.1"
CURRENT_SOURCE_NAME = "current_portfolio"
AUXILIARY_SOURCE_NAME = "early_stopping_min_delta"
AUXILIARY_ARCHIVE_COUNT = 600
CURRENT_SOURCE_LABEL = "80.1"
AUXILIARY_SOURCE_LABEL = "81.1"
PORTFOLIO_LABEL = "82.1"
SELECTION_RULE = (
    "per physical family, maximize guarded development paired F1 gain "
    "between independently audited 80.1 and preregistered 81.1 winners"
)


@contextmanager
def _configured_contract():
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "CURRENT_SOURCE_NAME": CURRENT_SOURCE_NAME,
        "AUXILIARY_SOURCE_NAME": AUXILIARY_SOURCE_NAME,
        "SOURCE_NAMES": {CURRENT_SOURCE_NAME, AUXILIARY_SOURCE_NAME},
        "AUXILIARY_ARCHIVE_COUNT": AUXILIARY_ARCHIVE_COUNT,
        "CURRENT_SOURCE_LABEL": CURRENT_SOURCE_LABEL,
        "AUXILIARY_SOURCE_LABEL": AUXILIARY_SOURCE_LABEL,
        "PORTFOLIO_LABEL": PORTFOLIO_LABEL,
        "SELECTION_RULE": SELECTION_RULE,
    }
    previous = {name: getattr(_base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(_base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(_base, name, value)


def _canonical_sha256(path: str | Path) -> str:
    return _base._canonical_sha256(path)


def _load_spec(path: str | Path) -> dict:
    with _configured_contract():
        return _base._load_spec(path)


def static_preflight(config_path: str | Path) -> dict:
    with _configured_contract():
        return _base.static_preflight(config_path)


def source_identity_preflight(config_path: str | Path) -> Path:
    with _configured_contract():
        return _base.source_identity_preflight(config_path)


def select(config_path: str | Path) -> Path:
    with _configured_contract():
        return _base.select(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=("static-preflight", "source-identity-preflight", "select"),
        required=True,
    )
    arguments = parser.parse_args()
    if arguments.mode == "static-preflight":
        static_preflight(arguments.config)
    elif arguments.mode == "source-identity-preflight":
        source_identity_preflight(arguments.config)
    else:
        select(arguments.config)


if __name__ == "__main__":
    main()
