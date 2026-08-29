# Verify_Task3_ResidualHeadDepthWidth_31.1

## Question

Can a systematic residual-head capacity search improve both the absolute FMT
classifier and its paired advantage over a same-width train-only Raw-PCA arm?

## Motivation and evidence boundary

The completed 25.1 semantic-block projection and 26.1 auxiliary-supervision
searches increased the development F1 gap to `0.19052` and `0.19201`, but their
absolute FMT F1 values fell to `0.88717` and `0.88490`. The strongest completed
absolute FMT result remains 22.1 at `0.88922`.

The earlier 6.1 architecture comparison tested one deep multilayer perceptron
cell with two hidden layers of width 64. It did not perform a complete
depth-by-width search within that winning architecture. Experiment 31.1 fills
that gap. It is an adaptive development experiment motivated by the completed
25.1/26.1 results; it is not presented as independently preregistered before
those results.

## Frozen protocol

- Development populations, labels, splits, frozen Raw checkpoints, training
  budget, and fusion selection come from 5.2.
- Each physical family uses the completed 22.1 anchored FMT feature.
- Layer Normalization, Gaussian Error Linear Unit activation, zero dropout,
  optimizer, loss, and paired seeds `40, 41, 42` remain fixed.
- FMT and train-only Raw-PCA use the same head width, depth, initialization,
  mini-batches, optimizer, and training budget in each candidate.
- Confirmation data remain closed.

## Complete factorial

Hidden width is one of `32, 48, 64, 80, 96`; hidden depth is one of `1, 2, 3`.
Their Cartesian product gives 15 candidates. Width 64 with depth 2 is the
exact historical control. The experiment contains 15 candidates, 10 data
entries, 3 paired seeds, and 2 arms: 900 training runs in 150 GPU array jobs.

Parameter counts differ across capacity cells by design, but the two arms are
identical within every cell. Preflight records all parameter counts and rejects
any candidate at or above the frozen Raw-wide cap of 148,225 parameters.

## Selection and targets

Selection is family-specific on exposed development data. A candidate is
eligible only when its absolute FMT F1 and Average Precision are both no lower
than the exact width-64/depth-2 control. Eligible candidates are ranked by
paired F1 gain, then absolute FMT F1 and the registered robustness metrics.

Joint development target:

- dataset-macro F1 gain over Raw-PCA at least `0.195`; and
- absolute dataset-macro FMT F1 at least `0.893`.

Failure of either requirement is a negative result. Any development winner
must still be evaluated on a fresh spatial population before supporting a
paper-level conclusion.

## Main files

- `config/Verify_Task3_ResidualHeadDepthWidth_31.1.yaml`
- `FMT_Utils/PathlineClassifier_3D.py`
- `Search_Task3_LossOptimization_7_1.py`
- `tests/test_task3_residual_head_depth_width_31_1.py`
- `ibex_bash/verify_task3_residual_head_depth_width_31.1_*.sh`

## Preflight result

The local full preflight failed before any Ibex submission or training result
was produced. Candidate `c14_w96_d3` exceeded the frozen Raw-wide parameter cap
for `smokeBuoyancy`. This is a valid capacity-guard failure, not a performance
result. Version 31.2 removes the complete width-96 level, retaining the
predeclared widths `32, 48, 64, 80` and all three depths as a complete 4-by-3
factorial. Version 31.1 and this failure record are retained.
