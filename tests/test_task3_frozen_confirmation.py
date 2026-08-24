import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Evaluate_Task3_FrozenConfirmation import _summarise
from Calibrate_Task3_ConstrainedAP import _select_constrained, _select_pareto


def _row(dataset, seed, variant, f1, ap):
    return {
        "dataset": dataset, "seed": seed, "variant": variant,
        "f1": f1, "average_precision": ap,
    }


def main():
    rows = [
        _row("flow", 20, "raw", 0.60, 0.70),
        _row("flow", 20, "raw_wide", 0.62, 0.68),
        _row("flow", 20, "raw_fmt_residual", 0.65, 0.73),
        _row("flow", 21, "raw", 0.61, 0.69),
        _row("flow", 21, "raw_wide", 0.60, 0.71),
        _row("flow", 21, "raw_fmt_residual", 0.64, 0.75),
    ]
    comparisons, aggregate = _summarise(rows, minimum_gain=0.02)
    assert len(comparisons) == 2 and len(aggregate) == 1
    assert np.isclose(comparisons[0]["gain_f1"], 0.03)
    assert np.isclose(comparisons[0]["gain_average_precision"], 0.03)
    assert np.isclose(aggregate[0]["mean_gain_f1"], 0.035)
    assert np.isclose(aggregate[0]["mean_gain_average_precision"], 0.045)
    assert np.isclose(
        aggregate[0]["gain_vs_per_seed_oracle_raw_f1"], 0.03
    )
    assert aggregate[0]["passes_minimum_gain"] == 1
    candidates = [
        {"alpha": 0.5, "validation_f1": 0.64,
         "validation_average_precision": 0.76},
        {"alpha": 1.0, "validation_f1": 0.67,
         "validation_average_precision": 0.74},
        {"alpha": 1.5, "validation_f1": 0.66,
         "validation_average_precision": 0.77},
    ]
    selected, count = _select_constrained(
        candidates, minimum_f1=0.65, minimum_ap=0.75
    )
    assert selected["alpha"] == 1.5 and count == 2
    pareto, count, maximum = _select_pareto(
        candidates, f1_tolerance=0.01
    )
    assert pareto["alpha"] == 1.5 and count == 2
    assert np.isclose(maximum, 0.67)
    print("TASK3 FROZEN CONFIRMATION TEST PASSED")


if __name__ == "__main__":
    main()
