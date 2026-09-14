#!/bin/bash
set -euo pipefail
cd /ibex/user/zhanx0o/FMT_AngleFeatures_20260914
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home/zhanx0o/anaconda3/envs/deepvortex/bin/python
PHASE="$1"
trap 'status=$?; "$PY" -u -m experiments.Record_Task1235_AngleFeatures_1_1 --phase "$PHASE" --event end --exit-code "$status"; exit "$status"' EXIT
"$PY" -u -m experiments.Record_Task1235_AngleFeatures_1_1 --phase "$PHASE" --event start
sha256sum --quiet -c SOURCE_MANIFEST.sha256
if [ "$PHASE" = "smoke" ]; then
    "$PY" tests/test_fmt_angles_3d.py -v
fi
"$PY" -u -m experiments.Run_Task1235_AngleFeatures_1_1 --phase "$PHASE" --index "${SLURM_ARRAY_TASK_ID:-0}"
if [ "$PHASE" = "audit" ]; then
    "$PY" -u -m experiments.Audit_Task1235_AngleFeatures_1_1 --root outputs/Verify_Task1235_AngleFeatures_1.1
fi
