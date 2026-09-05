#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT4B23
#SBATCH -o outputs/mainExp_Task4B_ChannelToTBL_2.3/tbl_GTs/logs/%x.%j.out
#SBATCH -e outputs/mainExp_Task4B_ChannelToTBL_2.3/tbl_GTs/logs/%x.%j.err
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G

set -euo pipefail

TASK4B_REPO_ROOT="${TASK4B_REPO_ROOT:-/ibex/scratch/zhanx0o/FMT_Task4B_ChannelToTBL_2_3/repo}"
cd "${TASK4B_REPO_ROOT}"

module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

hostname
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
sha256sum -c DEPLOYMENT_MANIFEST.sha256
python -m py_compile \
  FMT_Utils/Task4B_CrossFlow_3D.py \
  experiments/Train_Task4B_ChannelToTBL_2_1.py \
  experiments/Audit_Task4B_ChannelToTBL_2_1.py
python -m experiments.Train_Task4B_ChannelToTBL_2_1 \
  --config config/mainExp_Task4B_ChannelToTBL_2.3.yaml
python -m experiments.Audit_Task4B_ChannelToTBL_2_1 \
  --config config/mainExp_Task4B_ChannelToTBL_2.3.yaml

if find outputs/mainExp_Task4B_ChannelToTBL_2.3 -type f \
  \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \) \
  | grep -q .; then
  echo "ERROR: persistent checkpoint found" >&2
  exit 1
fi
