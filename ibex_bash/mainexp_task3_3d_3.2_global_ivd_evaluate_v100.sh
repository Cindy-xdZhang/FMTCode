#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT3g32eval
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=1:00:00
#SBATCH -p debug
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=v100
#SBATCH --mem=48G

set -euo pipefail
repo_root=${TASK3_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs
module load cuda/11.8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python Evaluate_Task3_MainTable.py \
  --config config/mainExp_Task3_3D_3.2_global_ivd_evaluate.yaml
