# Verify_Task3_ResidualOutputInitialization_30.1

## Question

Can preserving the frozen Raw classifier at residual-training initialization
improve both the absolute FMT classifier and its paired advantage over
train-only Raw-PCA?

## Rationale

The current residual branch ends in a randomly initialized linear layer. Its
initial nonzero logit therefore perturbs the frozen Raw classifier before the
FMT branch has learned a useful correction. A zero or small terminal weight
starts at, or close to, the exact frozen Raw prediction while retaining a
gradient path for learning the residual. This may particularly help the
low-dimensional anchored FMT features selected by 22.1.

This factor has not been searched by completed Task3 experiments. Experiment
30.1 was fixed without reading partial or final performance from 25.1--29.1,
so it is an independent development search rather than an adaptive response to
their intermediate outputs.

## Frozen protocol

- Development populations, labels, splits, frozen Raw checkpoints, optimizer,
  budget, early stopping, and fusion route come from 5.2.
- Each physical family uses the completed 22.1 anchored FMT feature.
- Both arms use the same width-64, depth-2 residual multilayer perceptron,
  paired seeds `40, 41, 42`, and candidate initialization.
- Confirmation data remain closed.

## Candidate grid

The exact PyTorch default is the control. Alternatives are terminal-weight
zero initialization; zero-mean normal initialization with standard deviation
`0.0001`, `0.001`, `0.005`, `0.01`, `0.025`, or `0.05`; and Xavier-uniform
initialization with gain `0.1`. Every non-default candidate sets terminal
residual-output biases to zero. For dual residual routes, both inference-time
output heads receive the same rule. Upstream projection and hidden layers are
not reinitialized.

Every cell is applied unchanged to FMT and train-only Raw-PCA. The search has
9 candidates, 10 data entries, 3 paired seeds, and 2 arms: 540 trainings in
90 array jobs. Initialization changes no parameter count.

## Selection and targets

A family-specific candidate is eligible only if its absolute FMT F1 and FMT
Average Precision are each no lower than the exact default control. Eligible
candidates are ranked by paired F1 gain, then absolute FMT F1 and the frozen
robustness tie-breakers.

Pre-registered joint target:

- dataset-macro F1 gain over Raw-PCA at least `0.195`; and
- absolute dataset-macro FMT F1 at least `0.893`.

Failure to reach either value is retained as a negative result. Any development
winner still requires a fresh spatial-population evaluation.

## Main files

- `config/Verify_Task3_ResidualOutputInitialization_30.1.yaml`
- `FMT_Utils/PathlineClassifier_3D.py`
- `Search_Task3_LossOptimization_7_1.py`
- `tests/test_task3_residual_output_initialization_30_1.py`
- `ibex_bash/verify_task3_residual_output_initialization_30.1_*.sh`
