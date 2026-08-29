# Verify_Task3_OverlapLoss_15.1

## Question

Can a paired soft Dice or Tversky overlap objective increase the supervised
Task3 FMT gain beyond the completed `Verify_Task3_LossOptimization_7.1`
development result (`+0.15064` dataset-macro F1 versus train-only Raw-PCA)?

## Frozen comparison

- Data, IVD labels, development split, frozen Raw classifier, family-specific
  pathline/FMT recipes, deep-MLP architecture, seeds `[40, 41, 42]`, and model
  selection protocol are inherited from the frozen 5.2/6.1 development setup.
- FMT and same-width train-only Raw-PCA receive exactly the same overlap loss
  candidate for each dataset and seed.
- The final/confirmation population remains unopened.

## Search

The 12 preregistered candidates include one control, four symmetric soft-Dice
weights, and seven Tversky variants. Tversky separately weights false
positives and false negatives; the recall-oriented variants assign 0.7 or
0.8 to false negatives. The full search is 10 datasets × 12 candidates × 3
seeds × 2 paired arms = 720 trainings in 120 GPU array children.

Primary selection is dataset-macro F1 gain over Raw-PCA. The preregistered
target is `+0.155`, which must exceed the completed 7.1 result. A failure or
negative result is retained in the experiment record and does not open
confirmation data.
