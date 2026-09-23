#!/usr/bin/env bash
# End-to-end Task 6 reproduction. Build the cache, verify it, train three seeds.
#
#   bash experiments/reproduce_task6.sh /path/to/Task6_CorelineDataset_2.7_share [/path/for/cache]
#
# Expected result: macro-F1 0.954 +- 0.002 (package split, single 1h shell).
# This script does NOT read any preintegrated/ directory. All streamlines are
# produced by the official frame.integrate(...) in task6_fields.py.
set -euo pipefail

ROOT="${1:?usage: reproduce_task6.sh <dataset root> [cache dir]}"
CACHE="${2:-$HOME/data/task6_icostar_r0250510}"
PY="${PYTHON:-python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

echo "=============================================================="
echo " dataset : $ROOT"
echo " cache   : $CACHE"
echo " python  : $($PY -c 'import sys;print(sys.version.split()[0])')"
echo "=============================================================="

[ -f "$ROOT/task6_fields.py" ] || { echo "ERROR: $ROOT does not look like the dataset package"; exit 1; }

echo
echo "[0/3] dataset identity"
$PY - "$ROOT" <<'EOS'
import sys, json, hashlib
from pathlib import Path
root = Path(sys.argv[1])
d = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
s = json.loads((root / 'default_split.json').read_text(encoding='utf-8'))
h = hashlib.sha256((root / 'dataset.json').read_bytes()).hexdigest()
print(f"  frames in dataset.json : {len(d['frames'])}   (expected 101)")
print(f"  split                  : train {len(s['train'])} / test {len(s['test'])}   (expected 77 / 24)")
print(f"  dataset.json sha256    : {h[:16]}...")
print(f"  reference sha256       : 6c0d28f95dc35724...  <- numbers below were measured on this")
if h[:16] != '6c0d28f95dc35724':
    print("  WARNING: different dataset revision. If the corelines were re-annotated the")
    print("           labels change and the reported numbers will not reproduce exactly.")
EOS

echo
echo "[1/3] building the star cache (8 shards, ~25 min, ~8 GB)"
mkdir -p "$CACHE"
for I in 0 1 2 3 4 5 6 7; do
  $PY experiments/Build_Task6_New_IcoStar_1_1.py \
      --root "$ROOT" --out "$CACHE" \
      --radii 0.25,0.5,1 --count 3000 \
      --threads 8 --shards 8 --shard "$I" &
done
wait

echo
echo "[2/3] verifying the cache"
$PY experiments/Check_Task6_Cache_1_1.py "$CACHE"
if ! $PY experiments/Check_Task6_Cache_1_1.py "$CACHE" | grep -q "CACHE MATCHES THE GUIDE"; then
  echo
  echo "ERROR: the cache does not match the reference configuration. Training would not"
  echo "       reproduce the reported numbers. Fix the differences listed above first."
  exit 1
fi

echo
echo "[3/3] training three seeds"
for S in 11 23 37; do
  $PY experiments/Run_Task6_NewStar_IcoVAE_1_1.py \
      --cache "$CACHE" --shells 1 \
      --lr 1e-3 --schedule cosine --warmup-steps 400 \
      --class-weight --steps 4000 --batch 512 \
      --seed "$S" --label "repro_seed$S" \
      --save-weights "outputs/weights_Task6/repro_seed$S.pt"
done

echo
echo "=============================================================="
$PY - <<'EOS'
import json, glob, numpy as np
f = [json.load(open(p))['best']['f1'] for p in sorted(glob.glob('outputs/exp_Task6_NewStar/repro_seed*.json'))]
if f:
    print(f"  reproduced macro-F1 : {np.mean(f):.4f} +- {np.std(f, ddof=1) if len(f) > 1 else 0:.4f}  (n={len(f)})")
    print(f"  reference           : 0.9549 +- 0.0018")
    ok = abs(np.mean(f) - 0.9549) < 0.01
    print(f"  {'REPRODUCED' if ok else 'MISMATCH - see the troubleshooting table in the guide'}")
EOS
echo "=============================================================="
