#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|v100
#SBATCH --time=02:00:00
#SBATCH -J T4B_Memo11
#SBATCH -o outputs/Verify_Task4B_FullVolumeMemorization_1.1/ibex_repro_20260903/logs/%x.%j.out
#SBATCH -e outputs/Verify_Task4B_FullVolumeMemorization_1.1/ibex_repro_20260903/logs/%x.%j.err

# Reproduction of Verify_Task4B_FullVolumeMemorization_1.1 on Ibex (GPU).
# Same frozen config and frozen cache; only the output directory is overridden.
# Nine runs (raw, fmt_only, raw_fmt x seeds 7068-7070), then the independent
# audit, then a hard check that no model checkpoint was persisted.

set -euo pipefail

PROJECT_ROOT="${FMT_PROJECT_ROOT:-/ibex/user/zhanx0o/FMT_Task4_Repro_20260903/FMT_Task4_Repro_20260903}"
OUT="outputs/Verify_Task4B_FullVolumeMemorization_1.1/ibex_repro_20260903"
CACHE="outputs/Other_Task4B_FMTFourClassClustering_1.2/channel_GTs/cache/task4b_four_class_cache.npz"

module load cuda/11.8 2>/dev/null || true
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

hostname
date -Is
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
printf 'deployment_base_commit='; cat DEPLOYMENT_BASE_COMMIT.txt
printf 'git_head='; git rev-parse HEAD
sha256sum -c DEPLOYMENT_MANIFEST.sha256
printf '%s  %s\n' 4be1dcedf69c3c5b4e2a47fd26fc46b2567db5dda9f547448e0cf6407af0a190 "$CACHE" | sha256sum -c
python -m py_compile \
  experiments/Verify_Task4B_FullVolumeMemorization_1_1.py \
  experiments/Audit_Task4B_FullVolumeMemorization_1_1.py \
  experiments/Train_Task4B_FourClassClassifier_1_1.py \
  FMT_Utils/Task4B_Classifier_3D.py \
  FMT_Utils/PathlineClassifier_3D.py

mkdir -p "$OUT/logs"
python -m experiments.Verify_Task4B_FullVolumeMemorization_1_1 \
  --config config/Verify_Task4B_FullVolumeMemorization_1.1.yaml \
  --output-dir "$OUT"
python -m experiments.Audit_Task4B_FullVolumeMemorization_1_1 \
  --config config/Verify_Task4B_FullVolumeMemorization_1.1.yaml \
  --output-dir "$OUT" \
  --write-json "$OUT/independent_audit.json"

if find "$OUT" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \) | grep -q .; then
  echo "ERROR: persistent checkpoint found" >&2
  exit 1
fi
sha256sum "$OUT/summary.json" "$OUT/independent_audit.json" "$OUT"/runs/*.json > "$OUT/evidence_sha256.txt"
cat "$OUT/evidence_sha256.txt"
date -Is
