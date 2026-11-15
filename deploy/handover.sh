#!/usr/bin/env bash
# Runs on an A6000 box (not on the Mac). Waits until this box has finished its own share of the queue, then
# starts a worker on the jobs handed over to it (stage 'hN-...' in queue/jobs.yaml, see deploy/plan_handover.py).
#   setsid -f nohup deploy/handover.sh <box-number> >> ~/lbb_handover.log 2>&1 < /dev/null
# One GPU job at a time: the first worker is stopped (queue/STOP, it is idle by then) before the second starts.
# The script changes nothing until the own share shows nothing to do, nothing running and nothing waiting on
# three checks in a row with no training process alive. It exits after starting the second worker.
set -uo pipefail
N=${1:?box number}
cd "${LBB_ROOT:-$HOME/label-budget-bench}"
POLL=${POLL:-300}
OWN='^([0-9]-|ablation)'
log() { echo "$(date -u +%FT%TZ) box$N $*"; }

left() {    # jobs of the own share not finished (claimable + running + waiting for labels); -1 if unreadable
  .venv/bin/python queue/worker.py --stage "$OWN" --shard "$N/8" --count 2>/dev/null | tail -n 1 |
    python3 -c 'import json,sys
try:
    d = json.loads(sys.stdin.read()); print(d["claimable"] + d["running"] + d["waiting_labels"])
except Exception:
    print(-1)'
}
workers() { ps -eo args | grep '[q]ueue/worker.py' | grep -v -e '--count' ; }
training() { ps -eo args | grep -q '[t]rain/run.py'; }

if ps -eo args | grep -q "[q]ueue/worker.py --stage ^h$N-"; then log "hand-over worker already running"; exit 0; fi
log "watching: own share has $(left) jobs left"
zero=0
while [ "$zero" -lt 3 ]; do
  sleep "$POLL"
  if [ "$(left)" = 0 ] && ! training; then zero=$((zero + 1)); else zero=0; fi
done

log "own share finished; stopping the first worker"
touch queue/STOP
while [ -n "$(workers)" ]; do sleep 10; done
while training; do sleep 30; done
rm -f queue/STOP
if command -v setsid > /dev/null; then
  setsid -f nohup .venv/bin/python queue/worker.py --stage "^h$N-" >> logs/worker_handover.log 2>&1 < /dev/null
else
  nohup .venv/bin/python queue/worker.py --stage "^h$N-" >> logs/worker_handover.log 2>&1 < /dev/null &
fi
sleep 5
log "started the hand-over worker: $(workers | head -n 1)"
