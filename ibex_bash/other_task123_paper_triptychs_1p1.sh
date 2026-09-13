#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-5%3
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd /home/zhanx0o/FMT_Uniform_3D_20260901
export PYTHONPATH="$PWD:$PWD/experiments"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
TASKS=(task1 task2 task3)
DATASETS=(cylinder3d tangaroa)
TASK="${TASKS[$((SLURM_ARRAY_TASK_ID / 2))]}"
DATASET="${DATASETS[$((SLURM_ARRAY_TASK_ID % 2))]}"
python visualization_1p1/Export_Task123_PaperTriptychs_1_1.py \
  --config visualization_1p1/Other_Task123_PaperTriptychs_1.1.json \
  --task "$TASK" --dataset "$DATASET"
