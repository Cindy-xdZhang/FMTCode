#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3ad531c
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G

set -euo pipefail

repo_root="${TASK531_REPO_ROOT:-/home/zhanx0o/FMT_Task3_AuxiliaryDropout_53_1}"
portfolio_root="${TASK354_REPO_ROOT:-/home/zhanx0o/FMT_Task3_ExtendedPortfolio_54_1}"
cd "$repo_root"
mkdir -p slurm_logs

output_root="$repo_root/outputs/Verify_Task3_AuxiliaryDropout_53.1"
candidate_root="$output_root/candidates"
evidence="$output_root/evidence_archive.json"
archive="$output_root/per_run_csv.tar.gz"
portfolio="$portfolio_root/outputs/Verify_Task3_ExtendedPortfolio_54.1/portfolio_selection.json"
portfolio_audit="$portfolio_root/outputs/Verify_Task3_ExtendedPortfolio_54.1/independent_audit.json"

resolved_outputs="$(realpath -e "$repo_root/outputs")"
resolved_output_root="$(realpath -e "$output_root")"
resolved_candidates="$(realpath -e "$candidate_root")"
expected_output_root="$resolved_outputs/Verify_Task3_AuxiliaryDropout_53.1"
expected_candidates="$expected_output_root/candidates"
if [[ "$resolved_output_root" != "$expected_output_root" ]] || \
   [[ "$resolved_candidates" != "$expected_candidates" ]]; then
  echo "refusing cleanup outside the exact resolved 53.1 output root: $resolved_candidates" >&2
  exit 1
fi

for required in "$evidence" "$archive" "$portfolio" "$portfolio_audit"; do
  test -s "$required"
done

python - "$evidence" "$archive" "$portfolio" "$portfolio_audit" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

evidence_path, archive_path, portfolio_path, audit_path = map(Path, sys.argv[1:])
evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if evidence.get("status") != "passed":
    raise SystemExit("53.1 evidence archive did not pass")
if int(evidence.get("archived_per_run_csv", -1)) != 660:
    raise SystemExit("53.1 archive does not cover all 660 per-run CSV files")
digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
if digest != str(evidence.get("stable_archive_sha256", "")):
    raise SystemExit("53.1 archive SHA-256 differs from its evidence record")
if audit.get("status") != "passed" or not audit.get("all_frozen_hashes_verified"):
    raise SystemExit("54.1 independent portfolio audit did not pass")
selected = {
    str(row.get("portfolio_source"))
    for row in dict(portfolio.get("primary_by_group", {})).values()
}
if "auxiliary_dropout" in selected:
    raise SystemExit("53.1 is selected by 54.1 and must not be deleted")
if int(portfolio.get("frozen_model_count", -1)) != 40:
    raise SystemExit("54.1 frozen portfolio is incomplete")
PY

checkpoint_count_before="$(find "$resolved_candidates" -type f \
  \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) | wc -l)"
if [[ "$checkpoint_count_before" -ne 660 ]]; then
  echo "expected exactly 660 disposable 53.1 checkpoints, found $checkpoint_count_before" >&2
  exit 1
fi

find "$resolved_candidates" -type f \
  \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) -delete

checkpoint_count_after="$(find "$resolved_candidates" -type f \
  \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) | wc -l)"
if [[ "$checkpoint_count_after" -ne 0 ]]; then
  echo "53.1 checkpoint cleanup incomplete: $checkpoint_count_after remain" >&2
  exit 1
fi

python - "$output_root/checkpoint_cleanup.json" \
  "$checkpoint_count_before" "$checkpoint_count_after" <<'PY'
import json
from pathlib import Path
import os
import sys

target = Path(sys.argv[1])
payload = {
    "experiment": "Verify_Task3_AuxiliaryDropout_53.1",
    "job_id": os.environ.get("SLURM_JOB_ID", ""),
    "checkpoint_count_before": int(sys.argv[2]),
    "checkpoint_count_after": int(sys.argv[3]),
    "reason": "not selected by audited 54.1 portfolio; CSV evidence retained",
    "status": "passed",
}
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "deleted_checkpoint_count=$checkpoint_count_before"
echo "remaining_checkpoint_count=$checkpoint_count_after"
