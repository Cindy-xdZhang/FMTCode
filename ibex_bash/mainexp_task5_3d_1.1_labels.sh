#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-3%4
#SBATCH -J FMTT5m11label
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=00:30:00
#SBATCH -p debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
repo_root=${TASK5_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
CONFIGS=(
  config/mainExp_Task5_3D_1.1_labels_development_old8.yaml
  config/mainExp_Task5_3D_1.1_labels_development_new2.yaml
  config/mainExp_Task5_3D_1.1_labels_confirmation_old8.yaml
  config/mainExp_Task5_3D_1.1_labels_confirmation_new2.yaml
)
python Build_Task3_GlobalIVD_Labels.py --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
