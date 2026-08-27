#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-99%24
#SBATCH -J FMTT3ivdr
#SBATCH -o slurm_logs/%x.%A_%a.out
#SBATCH -e slurm_logs/%x.%A_%a.err
#SBATCH --time=1:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint="a100|v100|rtx2080ti"
#SBATCH --mem=32G

set -euo pipefail
repo_root=${FMT_REPO_ROOT:-/home/zhanx0o/FMT_Task12_3D_20260823}
cd "$repo_root"
mkdir -p slurm_logs
module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

tags=(p80 p85 p87p5 p90 p92p5)
datasets=(cylinder3d halfcylinderRe640 halfcylinderRe6400 tangaroa \
          deltaWing_resampled deltaWing_LBM f22raptor channel \
          boeing747 smokeBuoyancy)
groups=(old8 old8 old8 old8 old8 old8 old8 old8 new2 new2)
pct_index=$((SLURM_ARRAY_TASK_ID / 20))
local_index=$((SLURM_ARRAY_TASK_ID % 20))
dataset_index=$((local_index % 10))
tag=${tags[$pct_index]}
dataset=${datasets[$dataset_index]}
group=${groups[$dataset_index]}
if (( local_index < 10 )); then
  mode=fmt
else
  mode=raw_pca
fi
config="outputs/Ablation_Task23IVDPercentile_1.1/generated_configs/task3_${tag}_${mode}_${group}.yaml"
output="outputs/Ablation_Task23IVDPercentile_1.1/task3/${tag}/development_${group}/${mode}_residual_shards/${dataset}"

nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
hostname
python Verify_Task3_FMTResidual.py \
  --config "$config" --dataset "$dataset" --output-dir "$output"
