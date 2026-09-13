#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --array=0-2%3
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd /home/zhanx0o/FMT_Uniform_3D_20260901
export PYTHONPATH="$PWD/objectivity_1p3:$PWD:$PWD/experiments"
export NATURE_FIGURE_SKILL_ROOT="$PWD/visualization_1p1"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
DATASETS=(cylinder3d halfcylinderRe640 halfcylinderRe6400)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
date --iso-8601=seconds
hostname
python -u objectivity_1p3/Visualize_FMT_Objectivity_Translation_1_3.py \
  --dataset "$DATASET" --global-mean --device cpu \
  --time-policy objectivity_1p3/cylinder_time_policy.json \
  --protocol-config config/mainExp_Task1_3D_4.1_uniform.yaml \
  --defer-pdf-collision-audit \
  --output-dir "outputs/Other_FMTObjectivityTranslation_1.3/$DATASET"
date --iso-8601=seconds
