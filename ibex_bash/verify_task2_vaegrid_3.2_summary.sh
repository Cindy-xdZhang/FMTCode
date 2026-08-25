#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT2vg32s
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
cd /home/zhanx0o/FMT_Task12_3D_20260823
mkdir -p slurm_logs
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
python Sweep_Task2_VAE_3D.py \
  --config config/Verify_Task2_VAEGrid_3D_3.2.yaml --summarize
python Build_Task2_VAE_Confirmation.py \
  --sweep-config config/Verify_Task2_VAEGrid_3D_3.2.yaml \
  --selection outputs/Verify_Task2_VAEGrid_3D_3.2/development_selection.json \
  --output config/mainExp_Task2_3D_3.3.yaml \
  --experiment mainExp_Task2_3D_3.3 \
  --output-dir outputs/mainExp_Task2_3D_3.3 \
  --confirmation-cache outputs/mainExp_Task3_3D_3.1/confirmation_cache_old8 \
  --newflow-confirmation-cache outputs/mainExp_Task3_3D_3.1/confirmation_cache_new2 \
  --confirmation-count 8 \
  --final-training-seeds 9068 9069 9070 9071 9072
