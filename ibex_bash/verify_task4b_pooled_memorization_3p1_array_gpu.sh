#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|v100|p100
#SBATCH --time=04:00:00
#SBATCH -J T4B_P31_MemoArr
#SBATCH --array=0-8
#SBATCH -o outputs/Verify_Task4B_PooledMemorization_3.1/pooled_sharded/logs/%x.%A_%a.out
#SBATCH -e outputs/Verify_Task4B_PooledMemorization_3.1/pooled_sharded/logs/%x.%A_%a.err

# Verify_Task4B_PooledMemorization_3.1 sharded: one (variant, seed) per array
# task, same frozen config, shared output directory.  Shards write
# summary_<variant>_seed<seed>.json; a CPU merge step assembles summary.json and
# the independent audit runs afterwards.

set -euo pipefail

PROJECT_ROOT="${FMT_PROJECT_ROOT:-/ibex/user/zhanx0o/FMT_Task4_Repro_20260903/FMT_Task4_Repro_20260903}"
CONFIG="config/Verify_Task4B_PooledMemorization_3.1.yaml"
OUT="outputs/Verify_Task4B_PooledMemorization_3.1/pooled_sharded"
CACHE="outputs/mainExp_Task4B_PooledInstanceSplit_3.1/pooled/cache/task4b_pooled_cache.npz"
VARIANTS=(raw fmt_only raw_fmt)
SEEDS=(7068 7069 7070)
INDEX="${SLURM_ARRAY_TASK_ID:?array task id required}"
VARIANT="${VARIANTS[$((INDEX / 3))]}"
SEED="${SEEDS[$((INDEX % 3))]}"

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
echo "shard index=$INDEX variant=$VARIANT seed=$SEED"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
printf 'git_head='; git rev-parse HEAD
sha256sum -c DEPLOYMENT_MANIFEST_Task4B_3p1_memo.sha256
printf '%s  %s\n' 40f21e358abb1999b5bf06bfa35db3f135d86ce8c9d245c4684e76618ef2d8a3 "$CACHE" | sha256sum -c

mkdir -p "$OUT/logs"
python -m experiments.Verify_Task4B_PooledMemorization_3_1 \
  --config "$CONFIG" --output-dir "$OUT" --variant "$VARIANT" --seed "$SEED"
date -Is
