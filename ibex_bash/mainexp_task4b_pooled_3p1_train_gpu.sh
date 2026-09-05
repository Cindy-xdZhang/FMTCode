#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|v100
#SBATCH --time=04:00:00
#SBATCH -J T4B_P31_Train
#SBATCH -o outputs/mainExp_Task4B_PooledInstanceSplit_3.1/pooled/logs/%x.%j.out
#SBATCH -e outputs/mainExp_Task4B_PooledInstanceSplit_3.1/pooled/logs/%x.%j.err

# mainExp_Task4B_PooledInstanceSplit_3.1, stage 2: train four variants x three
# seeds on the pooled cache (GPU), run the independent audit, and refuse any
# persisted model checkpoint.  Submitted with --dependency=afterok:<build job>.

set -euo pipefail

PROJECT_ROOT="${FMT_PROJECT_ROOT:-/ibex/user/zhanx0o/FMT_Task4_Repro_20260903/FMT_Task4_Repro_20260903}"
CONFIG="config/mainExp_Task4B_PooledInstanceSplit_3.1.yaml"
OUT="outputs/mainExp_Task4B_PooledInstanceSplit_3.1/pooled"

module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

hostname
date -Is
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
printf 'deployment_base_commit='; cat DEPLOYMENT_BASE_COMMIT.txt
printf 'git_head='; git rev-parse HEAD
sha256sum -c DEPLOYMENT_MANIFEST_Task4B_3p1.sha256
test -f "$OUT/cache/task4b_pooled_cache.npz"
sha256sum "$OUT/cache/task4b_pooled_cache.npz"

python -m experiments.Train_Task4B_PooledInstanceSplit_3_1 --config "$CONFIG"
python -m experiments.Audit_Task4B_PooledInstanceSplit_3_1 \
  --config "$CONFIG" \
  --output-dir "$OUT" \
  --write-json "$OUT/independent_audit.json"

if find "$OUT" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \) | grep -q .; then
  echo "ERROR: persistent checkpoint found" >&2
  exit 1
fi
sha256sum "$OUT/summary.json" "$OUT/independent_audit.json" "$OUT/per_run_metrics.csv" \
  "$OUT"/predictions/*.npz > "$OUT/evidence_sha256.txt"
cat "$OUT/evidence_sha256.txt"
date -Is
