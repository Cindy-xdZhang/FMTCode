#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-19%20
#SBATCH -J FMTT3g32res
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=1:30:00
#SBATCH -p debug
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint=v100
#SBATCH --mem=64G

set -euo pipefail
repo_root=${TASK3_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs

DATASETS=(
  cylinder3d halfcylinderRe640 halfcylinderRe6400 tangaroa
  deltaWing_resampled deltaWing_LBM f22raptor channel
  boeing747 smokeBuoyancy
)
DATA_GROUPS=(old8 old8 old8 old8 old8 old8 old8 old8 new2 new2)
base_index=$((SLURM_ARRAY_TASK_ID % 10))
dataset=${DATASETS[$base_index]}
data_group=${DATA_GROUPS[$base_index]}
if (( SLURM_ARRAY_TASK_ID < 10 )); then
  mode=fmt
else
  mode=raw_pca
fi
config="config/mainExp_Task3_3D_3.2_global_ivd_${mode}_${data_group}.yaml"
output_dir="outputs/mainExp_Task3_3D_3.2_global_ivd/development_${data_group}/${mode}_residual_shards/${dataset}"

if [[ "${TASK3_SHARD_DRY_RUN:-0}" == "1" ]]; then
  printf 'task=%s dataset=%s group=%s mode=%s config=%s output=%s\n' \
    "$SLURM_ARRAY_TASK_ID" "$dataset" "$data_group" "$mode" "$config" "$output_dir"
  exit 0
fi

module load cuda/11.8
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python Verify_Task3_FMTResidual.py \
  --config "$config" --dataset "$dataset" --output-dir "$output_dir"
