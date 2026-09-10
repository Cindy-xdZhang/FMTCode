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
  python -m experiments.Submit_Task678_DirectFMTFit_1_1 --config "$config" --event "$state" --phase "$phase" --exit-code "$code" || true
  exit "$code"
}
trap finish EXIT
python -m experiments.Submit_Task678_DirectFMTFit_1_1 --config "$config" --event RUNNING --phase "$phase"
case "$phase" in
  smoke)
    python -m unittest discover -s tests -p test_task678_directfmtfit_3d.py -v
    python -u tests/test_task678_directfmtfit_3d.py --smoke-output "outputs/Verify_Task678_DirectFitSmoke_1.1/job_${SLURM_JOB_ID}"
    ;;
  prepare) python -u -m experiments.Build_Task678_DirectFMTFit_1_1 --config "$config" --index "$SLURM_ARRAY_TASK_ID" --base-only ;;
  build) python -u -m experiments.Build_Task678_DirectFMTFit_1_1 --config "$config" --index "$SLURM_ARRAY_TASK_ID" ;;
  train_base|train_expanded) python -u -m experiments.Run_Task678_DirectFMTFit_1_1 --config "$config" --index "$SLURM_ARRAY_TASK_ID" ;;
  audit) python -u -m experiments.Audit_Task678_DirectFMTFit_1_1 --config "$config" ;;
  *) exit 2 ;;
esac
