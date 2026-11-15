#!/usr/bin/env bash
# Gate A crash test: run A uninterrupted; run B killed with SIGKILL after its first checkpoint,
# then restarted. Passes if B resumes from the checkpoint and its predictions match A's.
#   bash tests/crash_resume.sh                          # CPU smoke (ModernBERT-base, tiny)
#   bash tests/crash_resume.sh "--max-len 1024 --max-steps 60 --n 1000"   # GPU-sized
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
EXTRA=${1:-"--n 100 --max-len 128 --max-steps 12 --eval-limit 40"}
BASE="--family ${FAMILY:-encoder} --model ${MODEL:-modernbert_base} --task t2_response_safety --seed 0"
N=$(echo "$EXTRA" | sed -E 's/.*--n ([0-9]+).*/\1/')
ID="t2_response_safety__${MODEL:-modernbert_base}__gold__n${N}__s0"
rm -rf "preds/runs/${ID}__crashA" "preds/runs/${ID}__crashB" "checkpoints/${ID}__crashB"

$PY train/run.py $BASE $EXTRA --tag crashA > /tmp/crashA.log 2>&1
$PY train/run.py $BASE $EXTRA --tag crashB > /tmp/crashB1.log 2>&1 &
PID=$!
for _ in $(seq 1 1800); do
  [ -f "checkpoints/${ID}__crashB/state.pt" ] && break
  kill -0 $PID 2>/dev/null || { echo "FAIL: run B exited before writing a checkpoint"; exit 1; }
  sleep 1
done
kill -9 $PID
wait $PID 2>/dev/null || true
echo "killed run B (SIGKILL) after checkpoint"
$PY train/run.py $BASE $EXTRA --tag crashB > /tmp/crashB2.log 2>&1
grep -q "resumed from checkpoint" /tmp/crashB2.log || { echo "FAIL: run B did not resume"; exit 1; }
grep "resumed from checkpoint" /tmp/crashB2.log
$PY - "$ID" <<'EOF'
import sys, pandas as pd
i = sys.argv[1]
a = pd.read_parquet(f"preds/runs/{i}__crashA/preds.parquet")
b = pd.read_parquet(f"preds/runs/{i}__crashB/preds.parquet")
d = (a.p - b.p).abs().max()
print(f"max |p_A - p_B| = {d:.2e}")
sys.exit(0 if d < 1e-3 else 1)
EOF
echo "PASS: crash-resume"
