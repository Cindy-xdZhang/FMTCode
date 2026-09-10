#!/bin/bash
set -euo pipefail
cd /home/zhanx0o/FMT_nTDO_v2_20260910
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home/zhanx0o/anaconda3/envs/deepvortex/bin/python
PHASE="$1"
trap 'status=$?; "$PY" -u -m experiments.Record_Task1235_ObjectiveFMTnTDO_2_1 --phase "$PHASE" --event end --exit-code "$status"; exit "$status"' EXIT
"$PY" -u -m experiments.Record_Task1235_ObjectiveFMTnTDO_2_1 --phase "$PHASE" --event start
sha256sum --quiet -c SOURCE_MANIFEST.sha256
"$PY" -u -m experiments.Verify_Task1235_ObjectiveFMTnTDO_2_1 --phase "$1" --index "${SLURM_ARRAY_TASK_ID:-0}"
if [ "$1" = "audit" ]; then
    "$PY" -u -m experiments.Audit_Task1235_ObjectiveFMTnTDO_2_1 --root outputs/Verify_Task1235_ObjectiveFMTnTDO_2.1
fi
