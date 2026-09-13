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
export PYTHONPATH="$PWD/objectivity_1p3:$PWD:$PWD/experiments"
export NATURE_FIGURE_SKILL_ROOT="$PWD/visualization_1p1"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
date --iso-8601=seconds
hostname
python -u objectivity_1p3/Visualize_FMT_Objectivity_Translation_1_3.py \
  --dataset halfcylinderRe6400 --device cpu \
  --source-file objectivity_1p3/halfcylinderRe6400_late_window.nc \
  --observer-json objectivity_1p3/observer.json \
  --time-policy objectivity_1p3/cylinder_time_policy.json \
  --protocol-config config/mainExp_Task1_3D_4.1_uniform.yaml \
  --defer-pdf-collision-audit \
  --output-dir outputs/Other_FMTObjectivityTranslation_1.3/halfcylinderRe6400
date --iso-8601=seconds
