#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3apn751g
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G

set -euo pipefail
cd "${TASK375_REPO_ROOT:-/home/zhanx0o/FMT_Task3_AuxiliaryPostNorm_75_1}"
mkdir -p slurm_logs

source_root="${TASK374_REPO_ROOT:-/home/zhanx0o/FMT_Task3_AuxiliaryFeatureScalePortfolio_74_1}"
artifact_root="$source_root/outputs/Verify_Task3_AuxiliaryFeatureScalePortfolio_74.1"
selection="$artifact_root/portfolio_selection.json"
audit="$artifact_root/independent_audit.json"
test -s "$selection"
test -s "$audit"

original_state="$(sacct -n -X -j 51096759 --format=State -P | head -n 1)"
if [[ "$original_state" != "COMPLETED" ]]; then
  echo "74.1 independent audit job 51096759 is not COMPLETED: $original_state" >&2
  exit 1
fi

python - "$selection" "$audit" <<'PY'
import json
from pathlib import Path
import sys

selection = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
audit = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if selection.get("experiment") != "Verify_Task3_AuxiliaryFeatureScalePortfolio_74.1":
    raise SystemExit("unexpected 74.1 selection identity")
if int(selection.get("frozen_model_count", -1)) != 40 or int(
    selection.get("frozen_artifact_file_count", -1)
) != 80:
    raise SystemExit("74.1 frozen portfolio is incomplete")
if audit.get("status") != "passed" or not audit.get("all_frozen_hashes_verified"):
    raise SystemExit("74.1 independent audit did not pass")
if float(audit.get("maximum_absolute_difference_vs_portfolio", float("inf"))) != 0.0:
    raise SystemExit("74.1 independent audit differs from its selector")
PY

echo "original_audit_job=51096759"
echo "original_audit_state=$original_state"
echo "upstream_gate_status=passed"
