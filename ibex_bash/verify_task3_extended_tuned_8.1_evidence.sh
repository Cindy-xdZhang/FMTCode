#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3m81i
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail
cd "${TASK81_REPO_ROOT:-/home/zhanx0o/FMT_Task3_ExtendedTuned_8_1}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex

artifact_dir="outputs/mainExp_Task3_3D_8.1"
python Audit_Task3_ExtendedTuned_8_1.py \
  --config config/mainExp_Task3_3D_8.1.yaml \
  --artifact-dir "$artifact_dir" \
  --output "$artifact_dir/independent_audit.json"

test -s "$artifact_dir/per_run.csv"
test -s "$artifact_dir/summary.json"
test -s "$artifact_dir/frozen_recipe_manifest.json"
test -s "$artifact_dir/evaluation_preflight.json"
test -s "$artifact_dir/independent_audit.json"

sha256sum \
  "$artifact_dir/per_run.csv" \
  "$artifact_dir/summary.json" \
  "$artifact_dir/frozen_recipe_manifest.json" \
  "$artifact_dir/evaluation_preflight.json" \
  "$artifact_dir/independent_audit.json" \
  > "$artifact_dir/evidence_sha256.txt"
