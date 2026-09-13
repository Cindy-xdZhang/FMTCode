#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd /home/zhanx0o/FMT_Uniform_3D_20260901
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NATURE_FIGURE_SKILL_ROOT="$PWD/visualization_1p1"
python visualization_1p1/Visualize_Task123_PaperTriptychs_1_1.py "$@"
