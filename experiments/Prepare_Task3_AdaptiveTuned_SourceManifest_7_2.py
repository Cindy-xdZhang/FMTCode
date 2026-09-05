"""Derive the Task3 7.2 temporal-source manifest without copying flow data."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import Build_Task3_AdaptiveTuned_Confirmation_7_2 as spatial
import Prepare_Task3_FinalTuned_SourceManifest_7_1 as _base


@contextmanager
def _configured_base():
    previous = _base.spatial
    try:
        _base.spatial = spatial
        yield
    finally:
        _base.spatial = previous


def derive(config_path: str | Path, overwrite: bool = False) -> Path:
    with _configured_base():
        return _base.derive(config_path, overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    derive(args.config, args.overwrite)


if __name__ == "__main__":
    main()
