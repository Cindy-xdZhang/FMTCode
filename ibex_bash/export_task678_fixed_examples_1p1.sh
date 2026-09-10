#!/bin/bash
set -euo pipefail
REPORT_CHECKOUT="$1"
BASE_ROOT="$2"
VECTOR_ROOT="$3"
REPORT_ROOT="$4"
EXPECTED_COMMIT="$5"
cd "$REPORT_CHECKOUT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
source /home/zhanx0o/anaconda3/etc/profile.d/conda.sh
conda activate deepvortex
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
record_event() {
    python - "$BASE_ROOT/runtime_config.json" "$REPORT_ROOT" "$1" "$2" <<'PY'
import json,sys
from experiments.Submit_Task678_DirectFMTFit_1_1 import event
spec=json.load(open(sys.argv[1]));spec['output_root']=sys.argv[2]
event(spec,sys.argv[1],'fixed_examples_export',sys.argv[3],int(sys.argv[4]))
PY
}
record_event RUNNING 0
trap 'code=$?; if [ "$code" -ne 0 ]; then record_event FAILED "$code"; fi' EXIT
python -m experiments.Export_Task678_FixedExamples_1_1 --base "$BASE_ROOT" --vector "$VECTOR_ROOT" --output "$REPORT_ROOT/fixed_examples"
record_event COMPLETED 0
