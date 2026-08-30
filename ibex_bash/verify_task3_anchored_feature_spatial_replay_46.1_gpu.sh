#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-9%10
#SBATCH -J FMTT3r461
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=8G

set -euo pipefail
cd "${TASK461_REPO_ROOT:-/home/zhanx0o/FMT_Task3_AnchoredFeatureSpatialReplay_46_1}"
mkdir -p slurm_logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
python Replay_Task3_AnchoredFeatureSpatial_46_1.py \
  --config config/Verify_Task3_AnchoredFeatureSpatialReplay_46.1.yaml \
  --mode dataset --job-index "$SLURM_ARRAY_TASK_ID"
