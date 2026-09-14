#!/usr/bin/env bash
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
python -m experiments.Task4C_BaselineDevice_4_17 runtime --state STARTED
finish() {
    code=$?
    trap - EXIT
    python -m experiments.Task4C_BaselineDevice_4_17 runtime --state ENDED --exit-code "$code" || true
    exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python -m experiments.Task4C_BaselineDevice_4_17 verify
