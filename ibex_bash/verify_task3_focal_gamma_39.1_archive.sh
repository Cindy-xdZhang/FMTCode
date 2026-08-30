#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3fg391a
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail
cd "${TASK391_REPO_ROOT:-/home/zhanx0o/FMT_Task3_FocalGamma_39_1}"
mkdir -p slurm_logs

output_root="$(realpath outputs/Verify_Task3_FocalGamma_39.1)"
expected_root="$(realpath -m "$PWD/outputs/Verify_Task3_FocalGamma_39.1")"
if [[ "$output_root" != "$expected_root" ]]; then
  echo "refusing unexpected output root: $output_root" >&2
  exit 1
fi
candidate_root="$output_root/candidates"
for required in \
  "$output_root/optimization_leaderboard.csv" \
  "$output_root/optimization_selection.json" \
  "$output_root/preflight_manifest.json"; do
  test -s "$required"
done

per_run_count="$(find "$candidate_root" -type f -name per_run.csv | wc -l)"
if [[ "$per_run_count" -ne 720 ]]; then
  echo "expected 720 per-run CSV files, found $per_run_count" >&2
  exit 1
fi

archive="$output_root/per_run_csv.tar.gz"
tar \
  --exclude='*/checkpoints' \
  --exclude='*/checkpoints/*' \
  -czf "$archive" \
  -C "$output_root" candidates

archived_per_run_count="$(tar -tzf "$archive" | grep -c '/per_run.csv$')"
if [[ "$archived_per_run_count" -ne 720 ]]; then
  echo "archive contains $archived_per_run_count per-run CSV files" >&2
  exit 1
fi

(
  cd "$output_root"
  sha256sum \
    optimization_leaderboard.csv \
    optimization_selection.json \
    preflight_manifest.json \
    per_run_csv.tar.gz \
    > artifact_sha256.txt
)

mapfile -d '' checkpoints < <(
  find "$candidate_root" -type f \
    \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) -print0
)
deleted_count="${#checkpoints[@]}"
for checkpoint in "${checkpoints[@]}"; do
  case "$checkpoint" in
    "$candidate_root"/*/checkpoints/*)
      rm -f -- "$checkpoint"
      ;;
    *)
      echo "refusing checkpoint outside guarded root: $checkpoint" >&2
      exit 1
      ;;
  esac
done
find "$candidate_root" -depth -type d -name checkpoints -empty -delete

remaining_count="$(find "$candidate_root" -type f \
  \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) | wc -l)"
if [[ "$remaining_count" -ne 0 ]]; then
  echo "$remaining_count experiment checkpoints remain" >&2
  exit 1
fi

python -c 'import json, pathlib, sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({"archived_per_run_csv": int(sys.argv[2]), "deleted_checkpoint_count": int(sys.argv[3]), "remaining_checkpoint_count": int(sys.argv[4]), "status": "passed"}, indent=2, sort_keys=True), encoding="utf-8")' \
  "$output_root/checkpoint_cleanup.json" \
  "$archived_per_run_count" "$deleted_count" "$remaining_count"

echo "archived_per_run_csv=$archived_per_run_count"
echo "deleted_checkpoint_count=$deleted_count"
echo "remaining_checkpoint_count=$remaining_count"
