#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-6%7
#SBATCH -J FMTT23fig11
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=1:00:00
#SBATCH -p debug
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=v100
#SBATCH --mem=48G

set -euo pipefail
repo_root=${FMT_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs

module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

case "$SLURM_ARRAY_TASK_ID" in
  0) datasets=(channel) ;;
  1) datasets=(cylinder3d halfcylinderRe640 halfcylinderRe6400) ;;
  2) datasets=(tangaroa) ;;
  3) datasets=(deltaWing_resampled deltaWing_LBM) ;;
  4) datasets=(f22raptor) ;;
  5) datasets=(boeing747) ;;
  6) datasets=(smokeBuoyancy) ;;
  *) echo "invalid array index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python Visualize_Task23_3D_Horizontal.py \
  --tasks task2 task3 \
  --datasets "${datasets[@]}" \
  --predictions-only \
  --recompute
