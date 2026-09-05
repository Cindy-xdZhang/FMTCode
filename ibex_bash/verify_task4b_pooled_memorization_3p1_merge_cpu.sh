#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH -J T4B_P31_MemoMerge
#SBATCH -o outputs/Verify_Task4B_PooledMemorization_3.1/pooled_sharded/logs/%x.%j.out
#SBATCH -e outputs/Verify_Task4B_PooledMemorization_3.1/pooled_sharded/logs/%x.%j.err

# Merge the nine shard summaries, run the independent audit, refuse checkpoints.
# Submitted with --dependency=afterok:<array job>.

set -euo pipefail

PROJECT_ROOT="${FMT_PROJECT_ROOT:-/ibex/user/zhanx0o/FMT_Task4_Repro_20260903/FMT_Task4_Repro_20260903}"
CONFIG="config/Verify_Task4B_PooledMemorization_3.1.yaml"
OUT="outputs/Verify_Task4B_PooledMemorization_3.1/pooled_sharded"

source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

hostname; date -Is
sha256sum -c DEPLOYMENT_MANIFEST_Task4B_3p1_memo.sha256
ls "$OUT"/summary_*_seed*.json | wc -l
python -m experiments.Merge_Task4B_PooledMemorization_3_1_Shards --config "$CONFIG" --output-dir "$OUT"
python -m experiments.Audit_Task4B_PooledMemorization_3_1 \
  --config "$CONFIG" --output-dir "$OUT" --write-json "$OUT/independent_audit.json"
if find "$OUT" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \) | grep -q .; then
  echo "ERROR: persistent checkpoint found" >&2; exit 1
fi
sha256sum "$OUT/summary.json" "$OUT/independent_audit.json" "$OUT"/runs/*.json "$OUT"/summary_*_seed*.json > "$OUT/evidence_sha256.txt"
cat "$OUT/evidence_sha256.txt"; date -Is
