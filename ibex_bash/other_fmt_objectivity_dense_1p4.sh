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
export PYTHONPATH="$PWD/objectivity_1p4:$PWD:$PWD/experiments"
export NATURE_FIGURE_SKILL_ROOT="$PWD/visualization_1p1"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
DATASETS=(cylinder3d halfcylinderRe640 halfcylinderRe6400)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
OBSERVER=(--global-mean)
if [[ "$DATASET" == halfcylinderRe6400 ]]; then
  OBSERVER=(--source-file objectivity_1p4/halfcylinderRe6400_long_window.nc --observer-json objectivity_1p4/observer.json)
fi
date --iso-8601=seconds
hostname
python -u objectivity_1p4/Visualize_FMT_Objectivity_Translation_1_4.py \
  --dataset "$DATASET" "${OBSERVER[@]}" --device cpu \
  --time-policy objectivity_1p4/cylinder_time_policy.json \
  --protocol-config config/mainExp_Task1_3D_4.1_uniform.yaml \
  --defer-pdf-collision-audit \
  --output-dir "outputs/Other_FMTObjectivityTranslation_1.4/$DATASET"
date --iso-8601=seconds
