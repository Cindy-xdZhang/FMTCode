#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3rhb1p1081e
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail
cd "${TASK3108_REPO_ROOT:-/home/zhanx0o/FMT_Task3_ResidualHeadBeta1RefreshPortfolio_108_1}"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex

artifact_dir="outputs/Verify_Task3_ResidualHeadBeta1RefreshPortfolio_108.1"
python Audit_Task3_ResidualHeadBeta1RefreshPortfolio_108_1.py \
  --config config/Verify_Task3_ResidualHeadBeta1RefreshPortfolio_108.1.yaml \
  --artifact-dir "$artifact_dir" \
  --output "$artifact_dir/independent_audit.json"

result_count="$(find "$artifact_dir/frozen_artifacts" -type f -name per_run.csv | wc -l)"
checkpoint_count="$(find "$artifact_dir/frozen_artifacts" -type f \
  \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) | wc -l)"
total_count="$(find "$artifact_dir/frozen_artifacts" -type f | wc -l)"
if [[ "$result_count" -ne 40 || "$checkpoint_count" -ne 40 \
      || "$total_count" -ne 80 ]]; then
  echo "unexpected 108.1 frozen counts: results=$result_count checkpoints=$checkpoint_count total=$total_count" >&2
  exit 1
fi

echo "frozen_result_count=$result_count"
echo "frozen_checkpoint_count=$checkpoint_count"
echo "frozen_total_count=$total_count"
