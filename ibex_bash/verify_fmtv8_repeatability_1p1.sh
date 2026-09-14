#!/usr/bin/env bash
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
config=config/Ablation_FMTv8_Search_2.1.json
python -m experiments.FMTv8_Search_2_1 runtime --config "$config" --runtime-phase diagnostic --state STARTED
finish() {
    code=$?
    trap - EXIT
    python -m experiments.FMTv8_Search_2_1 runtime --config "$config" --runtime-phase diagnostic --state ENDED --exit-code "$code" || true
    exit "$code"
}
trap finish EXIT
python -m experiments.Verify_FMTv8_Repeatability_1_1
