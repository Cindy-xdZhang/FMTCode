#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-125%12
#SBATCH -J FMTT2vg31v
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=2:00:00
#SBATCH -p gpu4
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=v100
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
group_index=$((SLURM_ARRAY_TASK_ID / 18))
variant_index=$((18 + SLURM_ARRAY_TASK_ID % 18))
group_name=${TASK2_GROUPS[$group_index]}
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
printf 'group=%s variant_index=%s\n' "$group_name" "$variant_index"
python Sweep_Task2_VAE_3D.py \
  --config config/Verify_Task2_VAEGrid_3D_3.1.yaml \
  --group "$group_name" --variant-index "$variant_index" --resume
