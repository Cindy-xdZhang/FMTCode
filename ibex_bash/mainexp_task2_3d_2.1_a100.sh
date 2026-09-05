#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-5%6
#SBATCH -J FMTT2m21
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --constraint=a100
#SBATCH --mem=64G

set -euo pipefail
cd /home/zhanx0o/FMT_Task12_3D_20260823
mkdir -p slurm_logs
module load cuda/11.8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
TASK2_GROUPS=(channel halfcylinder_low halfcylinder_high tangaroa deltaWing f22raptor)
GROUP_NAME=${TASK2_GROUPS[$SLURM_ARRAY_TASK_ID]}
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
echo "Task2 config group: ${GROUP_NAME}"
python -m experiments.Run_Task2_3D_Main \
  --config config/mainExp_Task2_3D_2.1.yaml \
  --group "${GROUP_NAME}" --resume
