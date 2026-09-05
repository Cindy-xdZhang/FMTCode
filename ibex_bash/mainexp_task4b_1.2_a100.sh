#!/bin/bash
#SBATCH -N 1
#SBATCH -J FMTT4B12
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G

set -euo pipefail

TASK4B_REPO_ROOT="${TASK4B_REPO_ROOT:-/home/zhanx0o/FMT_Task4B_1_2}"
cd "${TASK4B_REPO_ROOT}"

module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

hostname
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
printf 'deployment_base_commit='
cat DEPLOYMENT_BASE_COMMIT.txt
sha256sum -c DEPLOYMENT_MANIFEST.sha256
python -m experiments.Preflight_Task4B_Supervised_1_2 \
  --config config/mainExp_Task4B_1.2.yaml \
  --require-cuda
python -m experiments.Train_Task4B_FourClassClassifier_1_1 \
  --config config/mainExp_Task4B_1.2.yaml

if find outputs/mainExp_Task4B_1.2 -type f \
  \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) | grep -q .; then
  echo "ERROR: persistent checkpoint found" >&2
  exit 1
fi
