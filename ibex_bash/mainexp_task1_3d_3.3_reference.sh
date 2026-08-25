#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-1%2
#SBATCH -J FMTT1r33
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=1:00:00
#SBATCH -p gpu4
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint="a100|v100"
#SBATCH --mem=48G

set -euo pipefail
cd /home/zhanx0o/FMT_Task12_3D_20260823
mkdir -p slurm_logs
module load cuda/11.8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
CONFIGS=(config/mainExp_Task1_3D_3.3_reference_old8.yaml config/mainExp_Task1_3D_3.3_reference_new2.yaml)
python Run_Task1_3D_Main.py --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
