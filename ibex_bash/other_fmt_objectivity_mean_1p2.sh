#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd /home/zhanx0o/FMT_Uniform_3D_20260901
export PYTHONPATH="$PWD/objectivity_1p2:$PWD:$PWD/experiments"
export NATURE_FIGURE_SKILL_ROOT="$PWD/visualization_1p1"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
date --iso-8601=seconds
hostname
python -u objectivity_1p2/Visualize_FMT_Objectivity_Translation_1_2.py \
  --dataset cylinder3d --global-mean --device cpu \
  --protocol-config config/mainExp_Task1_3D_4.1_uniform.yaml \
  --output-dir outputs/Other_FMTObjectivityTranslation_1.2/cylinder3d
date --iso-8601=seconds
