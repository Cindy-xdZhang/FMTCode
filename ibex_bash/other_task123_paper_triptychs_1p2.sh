#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-11%4
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd /home/zhanx0o/FMT_Uniform_3D_20260901
export PYTHONPATH="$PWD:$PWD/experiments"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
TASKS=(task1 task2 task3)
DATASETS=(halfcylinderRe640 halfcylinderRe6400 boeing747 deltaWing_LBM)
TASK="${TASKS[$((SLURM_ARRAY_TASK_ID / 4))]}"
DATASET="${DATASETS[$((SLURM_ARRAY_TASK_ID % 4))]}"
python visualization_1p2/Export_Task123_PaperTriptychs_1_2.py \
  --config visualization_1p2/Other_Task123_PaperTriptychs_1.2.json \
  --task "$TASK" --dataset "$DATASET"
