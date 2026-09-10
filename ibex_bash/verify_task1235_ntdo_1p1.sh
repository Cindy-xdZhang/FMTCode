#!/bin/bash
set -euo pipefail
cd /home/zhanx0o/FMT_nTDO_20260910
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
sha256sum --quiet -c SOURCE_MANIFEST.sha256
PY=/home/zhanx0o/anaconda3/envs/deepvortex/bin/python
"$PY" -u -m experiments.Verify_Task1235_ObjectiveFMTnTDO_1_1 --phase "$1" --index "${SLURM_ARRAY_TASK_ID:-0}"
