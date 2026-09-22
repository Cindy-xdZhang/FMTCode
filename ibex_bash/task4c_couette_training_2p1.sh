#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
/home/zhanx0o/anaconda3/envs/deepvortex/bin/python -m experiments.Task4C_CouetteTraining_2_1 "$1" --index "${SLURM_ARRAY_TASK_ID:-0}"
