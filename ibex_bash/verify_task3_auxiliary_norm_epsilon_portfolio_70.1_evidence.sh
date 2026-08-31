#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3nep701e
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail
cd "${TASK370_REPO_ROOT:-/home/zhanx0o/FMT_Task3_AuxiliaryNormEpsilonPortfolio_70_1}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex

artifact_dir="outputs/Verify_Task3_AuxiliaryNormEpsilonPortfolio_70.1"
python Audit_Task3_AuxiliaryNormEpsilonPortfolio_70_1.py \
  --config config/Verify_Task3_AuxiliaryNormEpsilonPortfolio_70.1.yaml \
  --artifact-dir "$artifact_dir" \
  --output "$artifact_dir/independent_audit.json"

result_count="$(find "$artifact_dir/frozen_artifacts" -type f -name per_run.csv | wc -l)"
checkpoint_count="$(find "$artifact_dir/frozen_artifacts" -type f \
  \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) | wc -l)"
total_count="$(find "$artifact_dir/frozen_artifacts" -type f | wc -l)"
if [[ "$result_count" -ne 40 || "$checkpoint_count" -ne 40 \
      || "$total_count" -ne 80 ]]; then
  echo "unexpected 70.1 frozen counts: results=$result_count checkpoints=$checkpoint_count total=$total_count" >&2
  exit 1
fi

sha256sum \
  "$artifact_dir/portfolio_selection.json" \
  "$artifact_dir/independent_audit.json" \
  "$artifact_dir/source_identity_preflight.json" \
  > "$artifact_dir/evidence_sha256.txt"
