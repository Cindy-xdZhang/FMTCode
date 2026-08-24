#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-19%20
#SBATCH -J FMTT3m31rsv
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=2:00:00
#SBATCH -p debug
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint=v100
#SBATCH --mem=64G

set -euo pipefail
cd /home/zhanx0o/FMT_Task12_3D_20260823
mkdir -p slurm_logs
module load cuda/11.8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

DATASETS=(
  cylinder3d halfcylinderRe640 halfcylinderRe6400 tangaroa
  deltaWing_resampled deltaWing_LBM f22raptor channel
  boeing747 smokeBuoyancy
)
GROUPS=(old8 old8 old8 old8 old8 old8 old8 old8 new2 new2)

base_index=$((SLURM_ARRAY_TASK_ID % 10))
dataset=${DATASETS[$base_index]}
group=${GROUPS[$base_index]}
if (( SLURM_ARRAY_TASK_ID < 10 )); then
  mode=fmt
else
  mode=raw_pca
fi
config="config/mainExp_Task3_3D_3.1_${mode}_${group}.yaml"
output_dir="outputs/mainExp_Task3_3D_3.1/development_${group}/${mode}_residual_shards/${dataset}"

nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
printf 'task=%s dataset=%s group=%s mode=%s output=%s\n' \
  "$SLURM_ARRAY_TASK_ID" "$dataset" "$group" "$mode" "$output_dir"
python Verify_Task3_FMTResidual.py \
  --config "$config" --dataset "$dataset" --output-dir "$output_dir"
