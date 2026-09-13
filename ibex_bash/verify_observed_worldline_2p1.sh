#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --array=0-2%3
set -euo pipefail
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd /home/zhanx0o/FMT_Uniform_3D_20260901
export PYTHONPATH="$PWD/observed_2p1:$PWD:$PWD/experiments"
export NATURE_FIGURE_SKILL_ROOT="$PWD/visualization_1p1"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
DATASETS=(cylinder3d halfcylinderRe640 halfcylinderRe6400)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
date --iso-8601=seconds
hostname
python -u observed_2p1/Verify_FMT_Observed_Pathline_2_1.py --dataset "$DATASET" --device cpu --time-policy observed_2p1/cylinder_time_policy.json --output-dir "outputs/Verify_FMTObservedPathline_2.1/$DATASET"
date --iso-8601=seconds
