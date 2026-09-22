"""Rebuild the Task4-c bundle cache with the relaxed neighbour head rule.

Rebinds `Task4C_PhysicalLength_4_14.center_and_neighbors` to the relaxed variant
at runtime and then runs the frozen `prepare` pipeline unchanged, so no existing
file is modified and every other rule, threshold and random draw is untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import Task4C_PhysicalLength_4_14 as builder  # noqa: E402
from FMT_Utils.Task4C_RelaxedHead_1_1 import center_and_neighbors_relaxed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/exp_Task4C_RelaxedHead_1.0.json")
    parser.add_argument("--index", type=int, required=True, help="0 = channel, 1 = tbl")
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()

    if builder.center_and_neighbors.__name__ != "center_and_neighbors":
        raise RuntimeError("unexpected seeding function already bound")
    builder.center_and_neighbors = center_and_neighbors_relaxed
    spec = builder.load_spec(arguments.config)
    print(f"seeding with {builder.center_and_neighbors.__name__} "
          f"(neighbour head-component test removed; centre unchanged)", flush=True)
    builder.prepare(spec, arguments.config, arguments.index, arguments.preflight, None)


if __name__ == "__main__":
    main()
