#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-1%2
#SBATCH -J FMTT3m31b
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint=a100
#SBATCH --mem=64G

set -euo pipefail
cd /home/zhanx0o/FMT_Task12_3D_20260823
mkdir -p slurm_logs
module load cuda/11.8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
CONFIGS=(
  config/mainExp_Task3_3D_3.1_baselines_old8.yaml
  config/mainExp_Task3_3D_3.1_baselines_new2.yaml
)
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python Verify_Task3_FMTClassifier.py --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
