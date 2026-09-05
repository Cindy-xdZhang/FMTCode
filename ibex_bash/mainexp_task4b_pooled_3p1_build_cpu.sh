#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J T4B_P31_Build
#SBATCH -o outputs/mainExp_Task4B_PooledInstanceSplit_3.1/pooled/logs/%x.%j.out
#SBATCH -e outputs/mainExp_Task4B_PooledInstanceSplit_3.1/pooled/logs/%x.%j.err

# mainExp_Task4B_PooledInstanceSplit_3.1, stage 1: build the pooled cache (CPU).
# Channel streamlines are integrated from channel.vtk; TBL rows are read from the
# byte-hash-verified 2.3 target cache chunks already present on Ibex.

set -euo pipefail

PROJECT_ROOT="${FMT_PROJECT_ROOT:-/ibex/user/zhanx0o/FMT_Task4_Repro_20260903/FMT_Task4_Repro_20260903}"
TBL_MANIFEST="${FMT_TBL_MANIFEST:-/ibex/user/zhanx0o/FMT_Task4B_ChannelToTBL_2_3/repo/outputs/mainExp_Task4B_ChannelToTBL_2.3/tbl_GTs/target_cache/target_cache_manifest.json}"
CONFIG="config/mainExp_Task4B_PooledInstanceSplit_3.1.yaml"

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

hostname
date -Is
printf 'deployment_base_commit='; cat DEPLOYMENT_BASE_COMMIT.txt
printf 'git_head='; git rev-parse HEAD
sha256sum -c DEPLOYMENT_MANIFEST_Task4B_3p1.sha256
python -m py_compile \
  FMT_Utils/Task4B_PooledSplit_3D.py \
  experiments/Build_Task4B_PooledInstanceSplit_3_1.py \
  experiments/Train_Task4B_PooledInstanceSplit_3_1.py \
  experiments/Audit_Task4B_PooledInstanceSplit_3_1.py
python tests/test_task4b_pooled_split_3d.py

python -m experiments.Build_Task4B_PooledInstanceSplit_3_1 \
  --config "$CONFIG" \
  --tbl-manifest "$TBL_MANIFEST"

sha256sum outputs/mainExp_Task4B_PooledInstanceSplit_3.1/pooled/cache/task4b_pooled_cache.npz \
  outputs/mainExp_Task4B_PooledInstanceSplit_3.1/pooled/cache/cache_summary.json
date -Is
