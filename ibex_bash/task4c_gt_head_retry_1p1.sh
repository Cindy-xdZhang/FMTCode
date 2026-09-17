#!/usr/bin/env bash
# Diagnostic wrapper only; the scientific trainer and configuration are frozen.
set -euo pipefail
export PYTHONFAULTHANDLER=1
cat /proc/self/limits
finish_retry() {
    code=$?
    trap - EXIT
    printf 'retry_wrapper_exit=%s\n' "$code"
    exit "$code"
}
trap finish_retry EXIT
bash ibex_bash/task4c_gt_head_coverage_1p1.sh "$@"
