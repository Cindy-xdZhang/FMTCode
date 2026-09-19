#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD" PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
PY=/home/zhanx0o/anaconda3/envs/deepvortex/bin/python
"$PY" -m unittest tests.test_task4c_ball_query_3_1
"$PY" -u -m experiments.Prepare_Task4C_BallQuery_3_1 --index "${SLURM_ARRAY_TASK_ID:?}"
