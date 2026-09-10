#!/bin/bash
set -euo pipefail
phase="$1"
kind="$2"
config="$3"
root="$4"
expected_commit="$5"
cd "${SLURM_SUBMIT_DIR:?}"
test "$(git rev-parse HEAD)" = "$expected_commit"
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
finish() {
  code=$?
  state=COMPLETED
  if [ "$code" -ne 0 ]; then state=FAILED; fi
  python -m experiments.Audit_Task678_PerDataset_1_1 --config "$config" --root "$root" --kind "$kind" --event "$state" --phase "$phase" --exit-code "$code" || true
  exit "$code"
}
trap finish EXIT
python -m experiments.Audit_Task678_PerDataset_1_1 --config "$config" --root "$root" --kind "$kind" --event RUNNING --phase "$phase"
if [ "$phase" = "audit_dataset" ]; then
  python -u -m experiments.Audit_Task678_PerDataset_1_1 --config "$config" --root "$root" --kind "$kind" --index "$SLURM_ARRAY_TASK_ID"
else
  python -u -m experiments.Audit_Task678_PerDataset_1_1 --config "$config" --root "$root" --kind "$kind" --collect
fi
