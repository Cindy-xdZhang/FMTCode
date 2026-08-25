#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-6%7
#SBATCH -J FMTT2m32
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=48G

set -euo pipefail
cd /home/zhanx0o/FMT_Task12_3D_20260823
mkdir -p slurm_logs
module load cuda/11.8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

TASK2_GROUPS=(channel halfcylinder tangaroa deltaWing f22raptor boeing747 smokeBuoyancy)
group_name=${TASK2_GROUPS[$SLURM_ARRAY_TASK_ID]}
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python Run_Task2_3D_Main.py \
  --config config/mainExp_Task2_3D_3.2.yaml --group "$group_name" --resume
