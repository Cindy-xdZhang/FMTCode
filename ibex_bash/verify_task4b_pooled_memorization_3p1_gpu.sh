#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|v100
#SBATCH --time=06:00:00
#SBATCH -J T4B_P31_Memo
#SBATCH -o outputs/Verify_Task4B_PooledMemorization_3.1/pooled/logs/%x.%j.out
#SBATCH -e outputs/Verify_Task4B_PooledMemorization_3.1/pooled/logs/%x.%j.err

# Verify_Task4B_PooledMemorization_3.1: fit = evaluation = all 91,711 rows of the
# pooled channel+TBL cache; Raw, FMT-only, Raw+FMT x seeds 7068-7070; PASS needs
# three consecutive zero-error epochs with positive minimum true-logit margin.
# Then the independent audit and a hard no-checkpoint check.

set -euo pipefail

PROJECT_ROOT="${FMT_PROJECT_ROOT:-/ibex/user/zhanx0o/FMT_Task4_Repro_20260903/FMT_Task4_Repro_20260903}"
CONFIG="config/Verify_Task4B_PooledMemorization_3.1.yaml"
OUT="outputs/Verify_Task4B_PooledMemorization_3.1/pooled"
CACHE="outputs/mainExp_Task4B_PooledInstanceSplit_3.1/pooled/cache/task4b_pooled_cache.npz"

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
sha256sum -c DEPLOYMENT_MANIFEST_Task4B_3p1_memo.sha256
printf '%s  %s\n' 40f21e358abb1999b5bf06bfa35db3f135d86ce8c9d245c4684e76618ef2d8a3 "$CACHE" | sha256sum -c

mkdir -p "$OUT/logs"
python -m experiments.Verify_Task4B_PooledMemorization_3_1 --config "$CONFIG"
python -m experiments.Audit_Task4B_PooledMemorization_3_1 \
  --config "$CONFIG" \
  --output-dir "$OUT" \
  --write-json "$OUT/independent_audit.json"

if find "$OUT" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \) | grep -q .; then
  echo "ERROR: persistent checkpoint found" >&2
  exit 1
fi
sha256sum "$OUT/summary.json" "$OUT/independent_audit.json" "$OUT"/runs/*.json > "$OUT/evidence_sha256.txt"
cat "$OUT/evidence_sha256.txt"
date -Is
