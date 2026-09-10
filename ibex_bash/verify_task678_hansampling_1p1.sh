#!/bin/bash
set -euo pipefail
phase="$1"
config="$2"
expected_commit="$3"
cd "${SLURM_SUBMIT_DIR:?}"
test "$(git rev-parse HEAD)" = "$expected_commit"
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONPATH="$PWD"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
finish() {
  code=$?
  state=COMPLETED
  if [ "$code" -ne 0 ]; then state=FAILED; fi
  python -m experiments.Submit_Task678_HanSampling_1_1 --config "$config" --event "$state" --phase "$phase" --exit-code "$code" || true
  exit "$code"
}
trap finish EXIT
python -m experiments.Submit_Task678_HanSampling_1_1 --config "$config" --event RUNNING --phase "$phase"
case "$phase" in
  smoke)
    python -m unittest tests.test_task678_hansampling_3d -v
    python -u -m tests.test_task678_hansampling_3d --smoke-output "outputs/Verify_Task678_HanSmoke_1.1/job_${SLURM_JOB_ID}"
    ;;
  build) python -u -m experiments.Build_Task678_HanSampling_1_1 --config "$config" --index "$SLURM_ARRAY_TASK_ID" ;;
  data_audit) python -u -m experiments.Audit_Task678_HanSampling_1_1 --config "$config" --data-only ;;
  train) python -u -m experiments.Run_Task678_HanSampling_1_1 --config "$config" --index "$SLURM_ARRAY_TASK_ID" ;;
  affine) python -u -m experiments.Run_Task678_HanSampling_1_1 --config "$config" --index "$SLURM_ARRAY_TASK_ID" --affine ;;
  audit) python -u -m experiments.Audit_Task678_HanSampling_1_1 --config "$config" ;;
  *) exit 2 ;;
esac
