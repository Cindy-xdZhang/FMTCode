#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
echo "AUDIT START $(date -Is) job=$SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID node=$(hostname)"
trap 'code=$?; echo "AUDIT END $(date -Is) exit=$code"; exit "$code"' EXIT
/home/zhanx0o/anaconda3/envs/deepvortex/bin/python outputs/Verify_Task6_DirectAudit_1.1/audit.py --config outputs/Verify_Task6_DirectAudit_1.1/config.json --index "$SLURM_ARRAY_TASK_ID"
