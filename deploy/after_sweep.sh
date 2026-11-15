#!/usr/bin/env bash
# Runs on an A6000 box. Waits until every sweep job of this box is finished (own share and handed-over jobs,
# and the hand-over watcher has exited), stops the idle worker, then runs extra work on the free GPU:
#   setsid -f nohup deploy/after_sweep.sh <box-number> <step>[,<step>...] >> ~/lbb_after_sweep.log 2>&1 < /dev/null
# Steps: detect:<detector>[+<detector>...]  (detect/run_detectors.py --all --only ...)   serve  (serve/bench.py)
# Gated detectors need HF_TOKEN in this script's environment (passed at start, never written to a file).
# Ends by writing logs/after_sweep_<box>.done. The copy loop keeps copying logs/ and preds/ to the storage server.
set -uo pipefail
N=${1:?box number}; STEPS=${2:?steps}
cd "${LBB_ROOT:-$HOME/label-budget-bench}"
POLL=${POLL:-300}
log() { echo "$(date -u +%FT%TZ) box$N $*"; }
left() {    # unfinished jobs for a stage pattern (+ shard); -1 if unreadable
  .venv/bin/python queue/worker.py --stage "$1" ${2:+--shard "$2"} --count 2>/dev/null | tail -n 1 |
    python3 -c 'import json,sys
try:
    d = json.loads(sys.stdin.read()); print(d["claimable"] + d["running"] + d["waiting_labels"])
except Exception:
    print(-1)'
}
training() { ps -eo args | grep -q '[t]rain/run.py'; }
watcher() { ps -eo args | grep -q '^bash deploy/[h]andover.sh'; }
workers() { ps -eo args | grep '[q]ueue/worker.py' | grep -v -e '--count'; }

log "waiting for the sweep on this box to finish; then: $STEPS (HF_TOKEN set: $([ -n "${HF_TOKEN:-}" ] && echo yes || echo no))"
zero=0
while [ "$zero" -lt 3 ]; do
  sleep "$POLL"
  if [ "$(left '^([0-9]-|ablation)' "$N/8")" = 0 ] && [ "$(left "^h$N-")" = 0 ] && ! training && ! watcher; then
    zero=$((zero + 1)); else zero=0; fi
done
log "sweep finished on this box; stopping the idle worker"
touch queue/STOP
while [ -n "$(workers)" ]; do sleep 10; done

IFS=, read -ra LIST <<< "$STEPS"
for s in "${LIST[@]}"; do
  log "start $s"
  case "$s" in
    detect:*) .venv/bin/python detect/run_detectors.py --all --only "$(echo "${s#detect:}" | tr + ,)" >> logs/detectors.log 2>&1 ;;
    serve)    .venv/bin/python serve/bench.py >> logs/serving.log 2>&1 ;;
    *)        log "unknown step $s"; continue ;;
  esac
  log "end $s rc=$?"
done
date -u +%FT%TZ > "logs/after_sweep_$N.done"
log "all steps finished"
